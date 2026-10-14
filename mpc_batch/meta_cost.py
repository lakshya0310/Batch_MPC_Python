"""
Multi-Modal Trajectory Ranking and Meta-Cost Evaluation.
Implements the multi-criteria ranking mechanism described in Section III-F
and `main_batch.cpp` of the paper.
"""

from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
import torch

from .optimizer import OptimizationResult


class MetaCostEvaluator:
    """
    Ranks batched candidate trajectories according to user-defined meta-costs.
    Supports:
    1. Cruise driving preference (v ~ v_cruise)
    2. Obstacle clearance / collision safety (||res_obs||_2)
    3. Right-lane preference (y ~ y_right_lane)
    4. High-speed driving preference (v ~ v_max)
    """
    def __init__(
        self,
        weights: List[float] = [50.0, 50.0, 0.0, 0.0],
        v_cruise: float = 15.0,
        v_max: float = 24.0,
        y_right_lane: float = -10.0,
        max_heading_deg: float = 15.0, # Heading limit from Section III (approx 13-15 degrees)
    ):
        self.weights = weights # [w_cruise, w_safety, w_right_lane, w_max_speed]
        self.v_cruise = v_cruise
        self.v_max = v_max
        self.y_right_lane = y_right_lane
        self.max_heading_rad = math.radians(max_heading_deg)

    def evaluate_and_rank(
        self,
        result: OptimizationResult,
    ) -> Tuple[int, np.ndarray, Dict[str, np.ndarray]]:
        """
        Evaluates raw meta-costs, ranks trajectories from 1 to L along each dimension,
        filters out trajectories with excessive heading changes, and selects the best candidate.
        
        Args:
            result: OptimizationResult from BatchTrajectoryOptimizer
            
        Returns:
            best_idx: Index of the optimal trajectory in the batch
            composite_costs: Array of composite scores for all goals [L]
            metrics_dict: Dictionary containing raw metrics and ranks
        """
        # Convert tensors to numpy arrays for ranking logic
        v = result.v.detach().cpu().numpy()     # [L, num]
        y = result.y.detach().cpu().numpy()     # [L, num]
        psi = result.psi.detach().cpu().numpy() # [L, num]
        res_obs = result.final_res_obs.detach().cpu().numpy() # [L]
        
        L = v.shape[0]
        
        # 1. Compute raw cost components
        cost_cruise = np.linalg.norm(v - self.v_cruise, axis=1) # [L]
        cost_safety = res_obs                                  # [L]
        cost_right_lane = np.linalg.norm(y - self.y_right_lane, axis=1) # [L]
        cost_max_speed = np.linalg.norm(v - self.v_max, axis=1) # [L]
        
        # 2. Check heading restriction (discard or heavily penalize large heading turns)
        max_heading_deviations = np.max(np.abs(psi), axis=1) # [L]
        heading_penalty = np.where(max_heading_deviations > self.max_heading_rad, 1000.0, 0.0)
        
        # 3. Compute ordinal ranks (1 to L) matching `get_ranks()` in main_batch.cpp
        # argsort gives rank order: 1 for lowest cost, L for highest cost
        rank_cruise = np.empty(L, dtype=np.float32)
        rank_safety = np.empty(L, dtype=np.float32)
        rank_right_lane = np.empty(L, dtype=np.float32)
        rank_max_speed = np.empty(L, dtype=np.float32)
        
        rank_cruise[np.argsort(cost_cruise)] = np.arange(1, L + 1)
        rank_safety[np.argsort(cost_safety)] = np.arange(1, L + 1)
        rank_right_lane[np.argsort(cost_right_lane)] = np.arange(1, L + 1)
        rank_max_speed[np.argsort(cost_max_speed)] = np.arange(1, L + 1)
        
        # 4. Composite weighted ranking score
        w0, w1, w2, w3 = self.weights
        composite_costs = (
            w0 * rank_cruise
            + w1 * rank_safety
            + w2 * rank_right_lane
            + w3 * rank_max_speed
            + heading_penalty
        )
        
        best_idx = int(np.argmin(composite_costs))
        
        metrics = {
            "cost_cruise": cost_cruise,
            "cost_safety": cost_safety,
            "cost_right_lane": cost_right_lane,
            "cost_max_speed": cost_max_speed,
            "rank_cruise": rank_cruise,
            "rank_safety": rank_safety,
            "rank_right_lane": rank_right_lane,
            "rank_max_speed": rank_max_speed,
            "composite_costs": composite_costs,
            "best_idx": best_idx,
        }
        
        return best_idx, composite_costs, metrics

