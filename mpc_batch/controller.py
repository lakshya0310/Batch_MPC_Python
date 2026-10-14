"""
Closed-Loop Model Predictive Control (MPC) Architecture.
Coordinates receding-horizon goal sampling, parallel batch optimization,
meta-cost ranking, vehicle kinematic integration, and highway traffic updates.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import math
import time
import numpy as np
import torch

from .bernstein import BernsteinBasis
from .optimizer import BatchTrajectoryOptimizer, OptimizerConfig, OptimizationResult
from .traffic import HighwayTrafficSimulation
from .meta_cost import MetaCostEvaluator


@dataclass
class EgoState:
    x: float = 0.0
    y: float = 10.0
    v: float = 14.0
    psi: float = 0.0
    psidot: float = 0.0


@dataclass
class StepTelemetry:
    step: int
    sim_time: float
    ego_state: EgoState
    control_v: float
    control_w: float
    best_goal_idx: int
    computation_time: float
    collision: bool
    result: OptimizationResult
    meta_metrics: Dict[str, Any]
    tracked_obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class BatchHighwayMPC:
    """
    Closed-loop Model Predictive Controller for Autonomous Highway Driving.
    Solves batch goal-directed optimizations at every cycle to achieve multi-modal driving.
    """
    def __init__(
        self,
        scenario: str = "cruise_IDM",
        config_path: Optional[str] = "config.yaml",
        custom_config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        self.scenario = scenario
        self.device = device
        
        # Load configuration
        import yaml
        if custom_config is not None:
            cfg = custom_config
        else:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                
        opt_cfg = cfg["optimizer"]
        sim_cfg = cfg["simulation"]
        scen_cfg = cfg["scenarios"].get(scenario, cfg["scenarios"]["cruise_IDM"])
        
        self.dt = sim_cfg["dt"]
        self.optimizer_config = OptimizerConfig(
            t_fin=opt_cfg["t_fin"],
            num=opt_cfg["num"],
            maxiter=opt_cfg["maxiter"],
            num_obs=opt_cfg["num_obs"],
            v_min=opt_cfg["v_min"],
            v_max=opt_cfg["v_max"],
            v_cruise=opt_cfg["v_cruise"],
            a_max=opt_cfg["a_max"],
            a_obs=opt_cfg["a_obs"],
            b_obs=opt_cfg["b_obs"],
            weight_smoothness=opt_cfg["weight_smoothness"],
            weight_smoothness_psi=opt_cfg["weight_smoothness_psi"],
            rho_init=opt_cfg["rho_init"],
            rho_scale=opt_cfg["rho_scale"],
            device=device,
        )
        
        self.optimizer = BatchTrajectoryOptimizer(self.optimizer_config)
        
        # Setup goal definitions
        self.goal_offsets_x = np.array(scen_cfg["x_g"], dtype=np.float32)
        self.goal_offsets_y = np.array(scen_cfg["y_g"], dtype=np.float32)
        self.num_goals = len(self.goal_offsets_x)
        
        # Meta-cost evaluator
        self.meta_evaluator = MetaCostEvaluator(
            weights=scen_cfg["weights"],
            v_cruise=opt_cfg["v_cruise"],
            v_max=opt_cfg["v_max"],
            y_right_lane=sim_cfg.get("rightmost_lane_y", -10.0),
        )
        
        # Simulation environment
        self.traffic_sim = HighwayTrafficSimulation(
            scenario=scenario,
            dt=self.dt,
            num_obs_tracked=opt_cfg["num_obs"],
            a_ell=opt_cfg["a_obs"],
            b_ell=opt_cfg["b_obs"],
            a_rect=sim_cfg["a_rect"],
            b_rect=sim_cfg["b_rect"],
        )
        
        # Ego state
        ego_init = scen_cfg.get("ego_init", {"x": 0.0, "y": 10.0, "v": 14.0, "psi": 0.0})
        self.ego = EgoState(
            x=ego_init["x"],
            y=ego_init["y"],
            v=ego_init["v"],
            psi=ego_init["psi"],
            psidot=0.0,
        )
        
        self.step_count = 0
        self.sim_time = 0.0
        self.telemetry_history: List[StepTelemetry] = []

    def reset(self):
        """Resets the MPC controller and traffic simulation."""
        self.traffic_sim.reset()
        self.step_count = 0
        self.sim_time = 0.0
        self.telemetry_history.clear()

    def step(self) -> StepTelemetry:
        """
        Executes one full MPC iteration:
        1. Formulates goal candidates relative to current position.
        2. Queries nearest obstacle telemetry.
        3. Runs batch trajectory optimization in PyTorch.
        4. Ranks trajectories and selects optimal plan.
        5. Applies control commands to ego vehicle.
        6. Updates surrounding traffic and collision status.
        """
        t_start = time.perf_counter()
        
        # 1. Sample goal positions (x relative to current ego position, y lane coordinates)
        x_goals = self.ego.x + self.goal_offsets_x
        y_goals = self.goal_offsets_y
        
        # 2. Query nearest 6 obstacles and their velocities
        x_obs, y_obs, vx_obs, vy_obs = self.traffic_sim.get_tracked_obstacles(self.ego.x, self.ego.y)
        
        # 3. Solve batch optimization
        res = self.optimizer.solve(
            x_init=self.ego.x,
            y_init=self.ego.y,
            v_init=self.ego.v,
            psi_init=self.ego.psi,
            psidot_init=self.ego.psidot,
            x_goals=x_goals,
            y_goals=y_goals,
            x_obs_init=x_obs,
            y_obs_init=y_obs,
            vx_obs=vx_obs,
            vy_obs=vy_obs,
        )
        
        # 4. Multi-modal ranking and optimal trajectory selection
        best_idx, composite_costs, metrics = self.meta_evaluator.evaluate_and_rank(res)
        
        t_comp = time.perf_counter() - t_start
        
        # 5. Extract smooth control commands from winning trajectory
        # Average of first 3 control steps (0.24s horizon) for smooth actuation matching C++
        control_v = float(res.v[best_idx, :3].mean().item())
        control_w = float(res.psidot[best_idx, :3].mean().item())
        
        # 6. Step ego vehicle kinematics forward
        prev_psi = self.ego.psi
        self.ego.psi += control_w * self.dt
        self.ego.psidot = (self.ego.psi - prev_psi) / self.dt
        self.ego.v = control_v
        
        vx = self.ego.v * math.cos(self.ego.psi)
        vy = self.ego.v * math.sin(self.ego.psi)
        self.ego.x += vx * self.dt
        self.ego.y += vy * self.dt
        
        # 7. Step surrounding IDM traffic forward
        self.traffic_sim.step_idm(self.ego.x, self.ego.y, self.ego.v)
        
        # Handle wrap-around for endless traffic
        lower_lim = self.ego.x - 30.0
        upper_lim = self.ego.x + 140.0
        self.traffic_sim.handle_wraparound(self.ego.x, upper_lim, lower_lim)
        
        # Check collision
        collision = self.traffic_sim.check_collision(self.ego.x, self.ego.y, self.ego.psi)
        
        self.step_count += 1
        self.sim_time += self.dt
        
        telemetry = StepTelemetry(
            step=self.step_count,
            sim_time=self.sim_time,
            ego_state=EgoState(
                x=self.ego.x,
                y=self.ego.y,
                v=self.ego.v,
                psi=self.ego.psi,
                psidot=self.ego.psidot,
            ),
            control_v=control_v,
            control_w=control_w,
            best_goal_idx=best_idx,
            computation_time=t_comp,
            collision=collision,
            result=res,
            meta_metrics=metrics,
            tracked_obs=(x_obs, y_obs, vx_obs, vy_obs),
        )
        self.telemetry_history.append(telemetry)
        
        return telemetry
