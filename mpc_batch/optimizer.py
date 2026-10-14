"""
Batch Non-Holonomic Trajectory Optimizer using PyTorch.
Implements the Alternating Minimization (Split-Bregman Augmented Lagrangian)
algorithm described in:
'Multi-Modal Model Predictive Control through Batch Non-Holonomic Trajectory Optimization: Application to Highway Driving'
Vivek K. Adajania et al. (IEEE RA-L 2022)

Key Novelty:
Parallelizes multiple goal-directed trajectory optimizations simultaneously across a batch dimension.
The KKT system matrices are identical across all batch instances, enabling one-shot batched linear solves.
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional
import torch
import numpy as np

from .bernstein import compute_bernstein, compute_bernstein_tensors, BernsteinBasis


@dataclass
class OptimizerConfig:
    t_fin: float = 8.0               # Planning horizon (s)
    num: int = 100                   # Discretization steps
    maxiter: int = 100               # AM iterations
    num_obs: int = 6                 # Number of obstacles
    v_min: float = 0.0               # Min velocity (m/s)
    v_max: float = 24.0              # Max velocity (m/s)
    v_cruise: float = 15.0           # Target cruise speed (m/s)
    a_max: float = 4.0               # Max total acceleration (m/s^2)
    a_obs: float = 5.6               # Collision ellipse semi-major axis (m)
    b_obs: float = 3.0               # Collision ellipse semi-minor axis (m)
    weight_smoothness: float = 2.5   # Smoothness weight for x, y
    weight_smoothness_psi: float = 5.0 # Smoothness weight for psi
    rho_init: float = 1.0            # Initial penalty parameter
    rho_scale: float = 1.06          # Penalty scaling factor per iteration
    device: str = "cpu"              # 'cpu' or 'cuda'
    dtype: torch.dtype = torch.float32


@dataclass
class OptimizationResult:
    x: torch.Tensor          # [L, num] X positions
    y: torch.Tensor          # [L, num] Y positions
    psi: torch.Tensor        # [L, num] Headings
    v: torch.Tensor          # [L, num] Forward velocities
    xdot: torch.Tensor       # [L, num] X velocities
    ydot: torch.Tensor       # [L, num] Y velocities
    xddot: torch.Tensor      # [L, num] X accelerations
    yddot: torch.Tensor      # [L, num] Y accelerations
    psidot: torch.Tensor     # [L, num] Yaw rates
    psiddot: torch.Tensor    # [L, num] Angular accelerations
    c_x: torch.Tensor        # [L, 11] Bernstein coefficients for x
    c_y: torch.Tensor        # [L, 11] Bernstein coefficients for y
    c_psi: torch.Tensor      # [L, 11] Bernstein coefficients for psi
    res_obs_history: torch.Tensor    # [maxiter] Obstacle residual norm
    res_bounds_history: torch.Tensor # [maxiter] Acceleration bounds residual norm
    res_nonhol_history: torch.Tensor # [maxiter] Non-holonomic residual norm
    res_psi_history: torch.Tensor    # [maxiter] Heading residual norm
    final_res_obs: torch.Tensor      # [L] Per-trajectory obstacle residual norm


class BatchTrajectoryOptimizer:
    """
    Batched Non-Holonomic Trajectory Optimizer for autonomous vehicles.
    Optimizes L trajectories in parallel with exact kinematic and collision avoidance constraints.
    """
    def __init__(self, config: Optional[OptimizerConfig] = None):
        if config is None:
            config = OptimizerConfig()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        self.dtype = config.dtype
        
        self.dt = config.t_fin / config.num
        self.tot_time = torch.linspace(0.0, config.t_fin, steps=config.num, device=self.device, dtype=self.dtype)
        
        # Precompute Bernstein basis matrices [num, 11]
        self.basis = compute_bernstein_tensors(self.tot_time, config.t_fin, device=self.device, dtype=self.dtype)
        self.P = self.basis.P
        self.Pdot = self.basis.Pdot
        self.Pddot = self.basis.Pddot
        self.nvar = self.P.shape[1] # 11
        
        # Higher resolution upsampled basis for control commands (dt = 0.008s)
        num_up = int(round(config.t_fin / 0.008))
        tot_time_up = torch.linspace(0.0, config.t_fin, steps=num_up, device=self.device, dtype=self.dtype)
        self.basis_up = compute_bernstein_tensors(tot_time_up, config.t_fin, device=self.device, dtype=self.dtype)
        self.Pdot_up = self.basis_up.Pdot
        
        # Smoothness cost matrices
        self.cost_smoothness = config.weight_smoothness * torch.matmul(self.Pddot.T, self.Pddot)
        self.cost_smoothness_psi = config.weight_smoothness_psi * torch.matmul(self.Pddot.T, self.Pddot)
        
        # Boundary constraint matrices
        # Position / Velocity: x(0), xdot(0), x(t_fin) -> 3 constraints
        self.A_eq = torch.stack([self.P[0, :], self.Pdot[0, :], self.P[-1, :]], dim=0) # [3, 11]
        
        # Heading: psi(0), psidot(0), psi(t_fin), psidot(t_fin) -> 4 constraints
        self.A_eq_psi = torch.stack([self.P[0, :], self.Pdot[0, :], self.P[-1, :], self.Pdot[-1, :]], dim=0) # [4, 11]
        
        # Constraint operators
        self.A_nonhol = self.Pdot      # [num, 11]
        self.A_psi = self.P            # [num, 11]
        self.A_acc = self.Pddot        # [num, 11]
        
        # A_obs is P stacked num_obs times vertically: [num * num_obs, 11] = [600, 11]
        self.A_obs = self.P.repeat(config.num_obs, 1)

    def _build_kkt_xy(
        self,
        rho_nonhol: float,
        rho_obs: float,
        rho_ineq: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds and factorizes the KKT matrix for (x, y) optimization.
        KKT matrix has shape [14, 14] and is identical across all batch goals.
        """
        cost = (
            self.cost_smoothness
            + rho_nonhol * torch.matmul(self.A_nonhol.T, self.A_nonhol)
            + rho_obs * torch.matmul(self.A_obs.T, self.A_obs)
            + rho_ineq * torch.matmul(self.A_acc.T, self.A_acc)
        ) # [11, 11]
        
        kkt = torch.zeros((self.nvar + 3, self.nvar + 3), device=self.device, dtype=self.dtype)
        kkt[:self.nvar, :self.nvar] = cost
        kkt[:self.nvar, self.nvar:] = self.A_eq.T
        kkt[self.nvar:, :self.nvar] = self.A_eq
        
        # Invert the 14x14 KKT matrix
        kkt_inv = torch.linalg.inv(kkt)
        return kkt, kkt_inv

    def _build_kkt_psi(self, rho_psi: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds and factorizes the KKT matrix for psi optimization.
        KKT matrix has shape [15, 15] and is identical across all batch goals.
        """
        cost_psi = self.cost_smoothness_psi + rho_psi * torch.matmul(self.A_psi.T, self.A_psi) # [11, 11]
        
        kkt_psi = torch.zeros((self.nvar + 4, self.nvar + 4), device=self.device, dtype=self.dtype)
        kkt_psi[:self.nvar, :self.nvar] = cost_psi
        kkt_psi[:self.nvar, self.nvar:] = self.A_eq_psi.T
        kkt_psi[self.nvar:, :self.nvar] = self.A_eq_psi
        
        kkt_psi_inv = torch.linalg.inv(kkt_psi)
        return kkt_psi, kkt_psi_inv

    def solve(
        self,
        x_init: float,
        y_init: float,
        v_init: float,
        psi_init: float,
        psidot_init: float,
        x_goals: torch.Tensor,
        y_goals: torch.Tensor,
        x_obs_init: torch.Tensor,
        y_obs_init: torch.Tensor,
        vx_obs: torch.Tensor,
        vy_obs: torch.Tensor,
        psi_fin: float = 0.0,
        psidot_fin: float = 0.0,
    ) -> OptimizationResult:
        """
        Solves the batch trajectory optimization problem for L goals.
        
        Args:
            x_init, y_init: Initial vehicle position (scalars)
            v_init: Initial forward velocity (scalar)
            psi_init: Initial heading angle (scalar, radians)
            psidot_init: Initial yaw rate (scalar, rad/s)
            x_goals: Target x coordinates of shape [L]
            y_goals: Target y coordinates of shape [L]
            x_obs_init: Initial obstacle x positions of shape [num_obs]
            y_obs_init: Initial obstacle y positions of shape [num_obs]
            vx_obs: Obstacle forward velocities of shape [num_obs]
            vy_obs: Obstacle lateral velocities of shape [num_obs]
            psi_fin: Target final heading (default 0.0)
            psidot_fin: Target final yaw rate (default 0.0)
            
        Returns:
            OptimizationResult containing trajectories, controls, and residuals.
        """
        # Ensure inputs are torch tensors on the correct device
        x_goals = torch.as_tensor(x_goals, device=self.device, dtype=self.dtype)
        y_goals = torch.as_tensor(y_goals, device=self.device, dtype=self.dtype)
        x_obs_init = torch.as_tensor(x_obs_init, device=self.device, dtype=self.dtype)
        y_obs_init = torch.as_tensor(y_obs_init, device=self.device, dtype=self.dtype)
        vx_obs = torch.as_tensor(vx_obs, device=self.device, dtype=self.dtype)
        vy_obs = torch.as_tensor(vy_obs, device=self.device, dtype=self.dtype)
        
        L = x_goals.shape[0] # Batch size (num_goal)
        num = self.config.num
        num_obs = self.config.num_obs
        
        # 1. Predict obstacle trajectories over planning horizon (Constant Velocity Extrapolation)
        # numbers: [num] steps from 0 to num-1
        t_steps = torch.arange(num, device=self.device, dtype=self.dtype) * self.dt
        # x_obs: [num_obs, num]
        x_obs = x_obs_init.unsqueeze(1) + vx_obs.unsqueeze(1) * t_steps.unsqueeze(0)
        y_obs = y_obs_init.unsqueeze(1) + vy_obs.unsqueeze(1) * t_steps.unsqueeze(0)
        
        # Flattened obstacle reference matching A_obs: [num_obs * num]
        # Obstacle 0 (all t), Obstacle 1 (all t), ..., Obstacle num_obs-1 (all t)
        x_obs_flat = x_obs.reshape(-1) # [600]
        y_obs_flat = y_obs.reshape(-1) # [600]
        
        # 2. Boundary conditions
        vx_init = v_init * math.cos(psi_init)
        vy_init = v_init * math.sin(psi_init)
        
        b_eq_x = torch.zeros((L, 3), device=self.device, dtype=self.dtype)
        b_eq_x[:, 0] = x_init
        b_eq_x[:, 1] = vx_init
        b_eq_x[:, 2] = x_goals
        
        b_eq_y = torch.zeros((L, 3), device=self.device, dtype=self.dtype)
        b_eq_y[:, 0] = y_init
        b_eq_y[:, 1] = vy_init
        b_eq_y[:, 2] = y_goals
        
        b_eq_psi = torch.zeros((L, 4), device=self.device, dtype=self.dtype)
        b_eq_psi[:, 0] = psi_init
        b_eq_psi[:, 1] = psidot_init
        b_eq_psi[:, 2] = psi_fin
        b_eq_psi[:, 3] = psidot_fin
        
        # 3. Variable initializations
        # Straight line guess between initial and goal positions for obstacle initialization
        interp_weights = torch.linspace(0.0, 1.0, steps=num, device=self.device, dtype=self.dtype).unsqueeze(0) # [1, num]
        x_guess = x_init + (x_goals.unsqueeze(1) - x_init) * interp_weights # [L, num]
        y_guess = y_init + (y_goals.unsqueeze(1) - y_init) * interp_weights # [L, num]
        
        # Compute initial obstacle variables alpha_obs and d_obs from initial guess
        # wc_alpha: x_guess - x_obs [L, num_obs * num]
        wc_alpha_init = (x_guess.unsqueeze(1) - x_obs.unsqueeze(0)).reshape(L, num_obs * num)
        ws_alpha_init = (y_guess.unsqueeze(1) - y_obs.unsqueeze(0)).reshape(L, num_obs * num)
        
        alpha_obs = torch.atan2(ws_alpha_init * self.config.a_obs, wc_alpha_init * self.config.b_obs)
        c1_d = self.config.rho_init * (
            (self.config.a_obs ** 2) * (torch.cos(alpha_obs) ** 2)
            + (self.config.b_obs ** 2) * (torch.sin(alpha_obs) ** 2)
        )
        c2_d = self.config.rho_init * (
            self.config.a_obs * wc_alpha_init * torch.cos(alpha_obs)
            + self.config.b_obs * ws_alpha_init * torch.sin(alpha_obs)
        )
        d_obs = torch.clamp(c2_d / c1_d, min=1.0)
        
        # State and auxiliary variables
        v = torch.full((L, num), v_init, device=self.device, dtype=self.dtype)
        psi = torch.full((L, num), psi_init, device=self.device, dtype=self.dtype)
        xdot = v * torch.cos(psi)
        ydot = v * torch.sin(psi)
        xddot = torch.zeros((L, num), device=self.device, dtype=self.dtype)
        yddot = torch.zeros((L, num), device=self.device, dtype=self.dtype)
        
        d_a = torch.full((L, num), self.config.a_max, device=self.device, dtype=self.dtype)
        alpha_a = torch.zeros((L, num), device=self.device, dtype=self.dtype)
        
        # Dual variables (Lagrange multipliers)
        lamda_x = torch.zeros((L, self.nvar), device=self.device, dtype=self.dtype)
        lamda_y = torch.zeros((L, self.nvar), device=self.device, dtype=self.dtype)
        lamda_psi = torch.zeros((L, self.nvar), device=self.device, dtype=self.dtype)
        
        # Penalty parameters
        rho_obs = self.config.rho_init
        rho_nonhol = self.config.rho_init
        rho_psi = self.config.rho_init
        rho_ineq = self.config.rho_init
        
        res_psi_history = torch.zeros(self.config.maxiter, device=self.device, dtype=self.dtype)
        res_obs_history = torch.zeros(self.config.maxiter, device=self.device, dtype=self.dtype)
        res_bounds_history = torch.zeros(self.config.maxiter, device=self.device, dtype=self.dtype)
        res_nonhol_history = torch.zeros(self.config.maxiter, device=self.device, dtype=self.dtype)
        
        c_x = torch.zeros((L, self.nvar), device=self.device, dtype=self.dtype)
        c_y = torch.zeros((L, self.nvar), device=self.device, dtype=self.dtype)
        c_psi = torch.zeros((L, self.nvar), device=self.device, dtype=self.dtype)
        
        # 4. Alternating Minimization Loop
        for it in range(self.config.maxiter):
            # A. Update psi (Eq. 13, 19, 20)
            psi_temp = torch.atan2(ydot, xdot) # [L, num]
            
            # Build and solve KKT for psi
            _, kkt_psi_inv = self._build_kkt_psi(rho_psi)
            
            # lincost_psi = -lamda_psi - rho_psi * (psi_temp @ A_psi)
            # A_psi is P of shape [num, nvar]
            lincost_psi = -lamda_psi - rho_psi * torch.matmul(psi_temp, self.A_psi) # [L, 11]
            rhs_psi = torch.cat([-lincost_psi, b_eq_psi], dim=1) # [L, 15]
            
            # One-shot batched solve for all goals in parallel!
            sol_psi = torch.matmul(rhs_psi, kkt_psi_inv.T) # [L, 15]
            c_psi = sol_psi[:, :self.nvar] # [L, 11]
            
            psi = torch.matmul(c_psi, self.P.T)       # [L, num]
            psidot = torch.matmul(c_psi, self.Pdot.T)  # [L, num]
            psiddot = torch.matmul(c_psi, self.Pddot.T)# [L, num]
            
            res_psi = psi - psi_temp # [L, num]
            lamda_psi = lamda_psi - rho_psi * torch.matmul(res_psi, self.A_psi)
            res_psi_history[it] = torch.norm(res_psi)
            
            # B. Update (x, y) (Eq. 12, 17, 18)
            _, kkt_xy_inv = self._build_kkt_xy(rho_nonhol, rho_obs, rho_ineq)
            
            # Kinematic target
            b_nonhol_x = v * torch.cos(psi) # [L, num]
            b_nonhol_y = v * torch.sin(psi) # [L, num]
            
            # Obstacle target
            b_obs_x = d_obs * torch.cos(alpha_obs) * self.config.a_obs + x_obs_flat.unsqueeze(0) # [L, 600]
            b_obs_y = d_obs * torch.sin(alpha_obs) * self.config.b_obs + y_obs_flat.unsqueeze(0) # [L, 600]
            
            # Acceleration target
            b_ax_ineq = d_a * torch.cos(alpha_a) # [L, num]
            b_ay_ineq = d_a * torch.sin(alpha_a) # [L, num]
            
            # Linear costs
            lincost_x = (
                -lamda_x
                - rho_nonhol * torch.matmul(b_nonhol_x, self.A_nonhol)
                - rho_obs * torch.matmul(b_obs_x, self.A_obs)
                - rho_ineq * torch.matmul(b_ax_ineq, self.A_acc)
            ) # [L, 11]
            
            lincost_y = (
                -lamda_y
                - rho_nonhol * torch.matmul(b_nonhol_y, self.A_nonhol)
                - rho_obs * torch.matmul(b_obs_y, self.A_obs)
                - rho_ineq * torch.matmul(b_ay_ineq, self.A_acc)
            ) # [L, 11]
            
            # RHS vectors: [-lincost, b_eq] of shape [L, 14]
            rhs_x = torch.cat([-lincost_x, b_eq_x], dim=1) # [L, 14]
            rhs_y = torch.cat([-lincost_y, b_eq_y], dim=1) # [L, 14]
            
            # One-shot batched solve for x and y across all goals simultaneously!
            sol_x = torch.matmul(rhs_x, kkt_xy_inv.T) # [L, 14]
            sol_y = torch.matmul(rhs_y, kkt_xy_inv.T) # [L, 14]
            
            c_x = sol_x[:, :self.nvar] # [L, 11]
            c_y = sol_y[:, :self.nvar] # [L, 11]
            
            x = torch.matmul(c_x, self.P.T)       # [L, num]
            xdot = torch.matmul(c_x, self.Pdot.T)  # [L, num]
            xddot = torch.matmul(c_x, self.Pddot.T)# [L, num]
            
            y = torch.matmul(c_y, self.P.T)       # [L, num]
            ydot = torch.matmul(c_y, self.Pdot.T)  # [L, num]
            yddot = torch.matmul(c_y, self.Pddot.T)# [L, num]
            
            # C. Update velocity v (Eq. 14, 21)
            v = torch.sqrt(xdot ** 2 + ydot ** 2)
            v = torch.clamp(v, min=self.config.v_min, max=self.config.v_max)
            v[:, 0] = v_init
            
            res_nonhol_x = xdot - v * torch.cos(psi) # [L, num]
            res_nonhol_y = ydot - v * torch.sin(psi) # [L, num]
            
            # D. Update obstacle avoidance variables alpha_obs and d_obs (Eq. 15, 22)
            wc_alpha = (x.unsqueeze(1) - x_obs.unsqueeze(0)).reshape(L, num_obs * num) # [L, 600]
            ws_alpha = (y.unsqueeze(1) - y_obs.unsqueeze(0)).reshape(L, num_obs * num) # [L, 600]
            
            alpha_obs = torch.atan2(ws_alpha * self.config.a_obs, wc_alpha * self.config.b_obs)
            
            c1_d = rho_obs * (
                (self.config.a_obs ** 2) * (torch.cos(alpha_obs) ** 2)
                + (self.config.b_obs ** 2) * (torch.sin(alpha_obs) ** 2)
            )
            c2_d = rho_obs * (
                self.config.a_obs * wc_alpha * torch.cos(alpha_obs)
                + self.config.b_obs * ws_alpha * torch.sin(alpha_obs)
            )
            d_obs = torch.clamp(c2_d / c1_d, min=1.0)
            
            res_x_obs = wc_alpha - self.config.a_obs * d_obs * torch.cos(alpha_obs)
            res_y_obs = ws_alpha - self.config.b_obs * d_obs * torch.sin(alpha_obs)
            
            # E. Update acceleration bound variables alpha_a and d_a (Eq. 16)
            alpha_a = torch.atan2(yddot, xddot) # [L, num]
            d_temp_a = torch.sqrt(xddot ** 2 + yddot ** 2)
            d_a = torch.clamp(d_temp_a, max=self.config.a_max)
            
            res_ax = xddot - d_a * torch.cos(alpha_a) # [L, num]
            res_ay = yddot - d_a * torch.sin(alpha_a) # [L, num]
            
            # F. Update Lagrange multipliers (Eq. 23)
            lamda_x = (
                lamda_x
                - rho_nonhol * torch.matmul(res_nonhol_x, self.A_nonhol)
                - rho_obs * torch.matmul(res_x_obs, self.A_obs)
                - rho_ineq * torch.matmul(res_ax, self.A_acc)
            )
            
            lamda_y = (
                lamda_y
                - rho_nonhol * torch.matmul(res_nonhol_y, self.A_nonhol)
                - rho_obs * torch.matmul(res_y_obs, self.A_obs)
                - rho_ineq * torch.matmul(res_ay, self.A_acc)
            )
            
            # Compute iteration residual norms for logging & convergence analysis
            res_obs_history[it] = torch.sqrt(torch.sum(res_x_obs ** 2 + res_y_obs ** 2))
            res_bounds_history[it] = torch.sqrt(torch.sum(res_ax ** 2 + res_ay ** 2))
            res_nonhol_history[it] = torch.sqrt(torch.sum(res_nonhol_x ** 2 + res_nonhol_y ** 2))
            
            # G. Penalty scaling (Split-Bregman parameter ramping)
            rho_obs *= self.config.rho_scale
            rho_nonhol *= self.config.rho_scale
            rho_psi *= self.config.rho_scale
            rho_ineq *= self.config.rho_scale
            
        final_res_obs = torch.sqrt(torch.sum(res_x_obs ** 2 + res_y_obs ** 2, dim=1)) # [L]
        
        return OptimizationResult(
            x=x,
            y=y,
            psi=psi,
            v=v,
            xdot=xdot,
            ydot=ydot,
            xddot=xddot,
            yddot=yddot,
            psidot=psidot,
            psiddot=psiddot,
            c_x=c_x,
            c_y=c_y,
            c_psi=c_psi,
            res_obs_history=res_obs_history,
            res_bounds_history=res_bounds_history,
            res_nonhol_history=res_nonhol_history,
            res_psi_history=res_psi_history,
            final_res_obs=final_res_obs,
        )
