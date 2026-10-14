"""
Highway Traffic Simulation and Intelligent Driver Model (IDM).
Simulates multi-lane highway traffic with dynamic surrounding vehicles,
IDM car-following dynamics, obstacle selection, and collision detection.
Faithfully reproduces `highway_car2.py` from the paper's original implementation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
import torch


@dataclass
class ObstacleVehicle:
    id: int
    x: float
    y: float
    vx: float
    vy: float = 0.0
    psi: float = 0.0
    v_desired: float = 10.0
    length: float = 4.0
    width: float = 1.4


class HighwayTrafficSimulation:
    """
    Simulates multi-lane highway traffic using the Intelligent Driver Model (IDM).
    Manages 18 surrounding vehicles across 6 lanes, performs obstacle selection (top 6 nearest),
    and implements vehicle wrap-around for endless highway simulation.
    """
    def __init__(
        self,
        scenario: str = "cruise_IDM",
        dt: float = 0.08,
        num_obs_tracked: int = 6,
        a_ell: float = 5.6,
        b_ell: float = 3.0,
        a_rect: float = 4.0,
        b_rect: float = 1.4,
    ):
        self.scenario = scenario
        self.dt = dt
        self.num_obs_tracked = num_obs_tracked
        self.a_ell = a_ell
        self.b_ell = b_ell
        self.a_rect = a_rect
        self.b_rect = b_rect
        
        self.lanes = [-10.0, -6.0, -2.0, 2.0, 6.0, 10.0]
        self.total_obs = 18
        
        # IDM Parameters matching highway_car2.py
        self.s0 = 1.0      # Minimum distance (m)
        self.T = 1.0       # Safe time headway (s)
        self.l_idm = a_ell # Effective vehicle spacing length (m)
        self.a_idm = 4.0   # Max acceleration (m/s^2)
        self.b_idm = 3.0   # Comfortable deceleration (m/s^2)
        
        self.reset()

    def reset(self):
        """Resets the simulation to initial conditions based on scenario."""
        # Setup 18 vehicles: 3 on each of the 6 lanes
        lane_y_repeated = self.lanes * 3 # 18 values
        
        if self.scenario in ["cruise_IDM", "HSRL_IDM"]:
            x_init = np.array([
                -10.0, 25.0, 60.0, -20.0, 40.0, 35.0,
                70.0, 85.0, 100.0, 10.0, 80.0, 170.0,
                130.0, 110.0, 160.0, 140.0, 135.0, 95.0
            ], dtype=np.float64)
        else: # RL_IDM
            x_init = np.array([
                -10.0, 25.0, 60.0, -20.0, 5.0, -35.0,
                90.0, 85.0, 75.0, 10.0, 25.0, 50.0,
                20.0, 40.0, 10.0, 100.0, 50.0, 95.0
            ], dtype=np.float64)
            
        desired_speeds = np.array([
            10.0, 11.0, 8.0, 9.5, 8.5, 12.0,
            7.5, 8.5, 7.0, 9.0, 8.5, 9.0,
            10.0, 9.5, 7.2, 9.5, 10.0, 10.5
        ], dtype=np.float64)
        
        # Initial speeds rolled by 6 matching C++/Python setup
        vx_init = np.roll(desired_speeds, 6).copy()
        
        self.vehicles = []
        for i in range(self.total_obs):
            self.vehicles.append(ObstacleVehicle(
                id=i,
                x=float(x_init[i]),
                y=float(lane_y_repeated[i]),
                vx=float(vx_init[i]),
                vy=0.0,
                psi=0.0,
                v_desired=float(desired_speeds[i]),
                length=self.a_rect,
                width=self.b_rect,
            ))
            
        self.ego_x = 0.0
        self.ego_y = 10.0
        self.ego_v = 14.0
        self.ego_psi = 0.0

    def step_idm(self, ego_x: float, ego_y: float, ego_v: float):
        """
        Updates speeds and positions of surrounding traffic according to IDM.
        Directly mirrors `IDM()` in `highway_car2.py`.
        """
        self.ego_x = ego_x
        self.ego_y = ego_y
        self.ego_v = ego_v
        
        # 1. Update velocities using IDM
        for i, veh in enumerate(self.vehicles):
            x1 = veh.x
            y1 = veh.y
            v1 = veh.vx
            v_des = veh.v_desired
            
            # Find nearest vehicle ahead in the same lane
            nearest_idx = -1
            min_dist = float("inf")
            for j, other in enumerate(self.vehicles):
                if i != j and abs(other.y - y1) < 0.1: # Same lane
                    if other.x > x1:
                        dist = other.x - x1
                        if dist < min_dist:
                            min_dist = dist
                            nearest_idx = j
                            
            v0 = 0.0
            x0 = 0.0
            has_lead = False
            
            if nearest_idx != -1:
                v0 = self.vehicles[nearest_idx].vx
                x0 = self.vehicles[nearest_idx].x
                has_lead = True
                
            # Check interaction with ego vehicle if ego is in the same lane ahead
            if abs(y1 - ego_y) <= self.b_ell and ego_x > x1:
                if has_lead:
                    if ego_x < x0:
                        v0 = ego_v
                        x0 = ego_x
                else:
                    v0 = ego_v
                    x0 = ego_x
                    has_lead = True
                    
            if has_lead:
                delta_v = v1 - v0
                s_alpha = x0 - x1 - self.l_idm
                if s_alpha <= 0.1:
                    s_alpha = 0.1
                s_star = self.s0 + v1 * self.T + (v1 * delta_v) / (2.0 * math.sqrt(self.a_idm * self.b_idm))
                acc = self.a_idm * (1.0 - (v1 / v_des) ** 4 - (s_star / s_alpha) ** 2)
                acc = max(-self.a_idm, acc)
            else:
                acc = self.a_idm * (1.0 - (v1 / v_des) ** 4)
                acc = min(self.a_idm, acc)
                
            veh.vx += acc * self.dt
            if veh.vx < 0.0:
                veh.vx = 0.0
                
        # 2. Advance positions
        for veh in self.vehicles:
            veh.x += veh.vx * self.dt
            veh.y += veh.vy * self.dt

    def handle_wraparound(self, ego_x: float, upper_lim: float, lower_lim: float):
        """
        Wraps vehicles that fall behind to the front of the traffic flow,
        ensuring an endless stream of highway vehicles.
        Matches lines 415-421 in `highway_car2.py`.
        """
        for i, veh in enumerate(self.vehicles):
            if veh.x < lower_lim - 6.0:
                veh.x = upper_lim + 5.0 + veh.vx
                veh.vx -= 2.0 * (i % 2)
                if (abs(veh.y - (-10.0)) < 0.1 and abs(veh.y - (-6.0)) < 0.1) or self.scenario in ["cruise_IDM", "RL_IDM"]:
                    veh.x += 15.0 * (i % 3)

    def get_tracked_obstacles(
        self,
        ego_x: float,
        ego_y: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects the top 6 nearest obstacles relevant to the ego vehicle.
        Uses the distance prioritization metric from `highway_car2.py` which penalizes vehicles behind ego.
        
        Returns:
            x_obs, y_obs, vx_obs, vy_obs of shape [num_obs_tracked]
        """
        scored_vehicles = []
        for veh in self.vehicles:
            dx = veh.x - ego_x
            dy = veh.y - ego_y
            base_dist = math.sqrt(dx ** 2 + dy ** 2)
            
            # Penalize vehicles that have already passed ego:
            # ((-dx - a_ell/2) / (|...| + 0.0001) + 1) * 10000
            diff = -dx - self.a_ell / 2.0
            penalty = (((diff) / (abs(diff) + 1e-4)) + 1.0) * 10000.0
            score = base_dist + penalty
            scored_vehicles.append((score, veh))
            
        # Sort by distance metric
        scored_vehicles.sort(key=lambda item: item[0])
        
        top_vehicles = [item[1] for item in scored_vehicles[:self.num_obs_tracked]]
        
        x_obs = np.array([v.x for v in top_vehicles], dtype=np.float32)
        y_obs = np.array([v.y for v in top_vehicles], dtype=np.float32)
        vx_obs = np.array([v.vx for v in top_vehicles], dtype=np.float32)
        vy_obs = np.array([v.vy for v in top_vehicles], dtype=np.float32)
        
        return x_obs, y_obs, vx_obs, vy_obs

    def check_collision(self, ego_x: float, ego_y: float, ego_psi: float = 0.0) -> bool:
        """
        Checks whether the ego vehicle collides with any surrounding obstacle.
        Uses the collision ellipse model defined in Eq. (1e) of the paper.
        """
        for veh in self.vehicles:
            dx = ego_x - veh.x
            dy = ego_y - veh.y
            # Ellipse collision metric
            val = (dx ** 2) / (self.a_ell ** 2) + (dy ** 2) / (self.b_ell ** 2)
            if val < 1.0:
                return True
        return False
