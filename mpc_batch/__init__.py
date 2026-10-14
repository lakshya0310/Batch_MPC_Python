"""
Batch Non-Holonomic Trajectory Optimization for Multi-Modal Highway Driving MPC.
PyTorch Implementation of the paper:
'Multi-Modal Model Predictive Control through Batch Non-Holonomic Trajectory Optimization: Application to Highway Driving'
Vivek K. Adajania, Aditya Sharma, Anish Gupta, Houman Masnavi, K Madhava Krishna, Arun K. Singh (IEEE RA-L 2022).
"""

from .bernstein import compute_bernstein, compute_bernstein_tensors, BernsteinBasis
from .optimizer import BatchTrajectoryOptimizer, OptimizerConfig, OptimizationResult
from .traffic import HighwayTrafficSimulation, ObstacleVehicle
from .meta_cost import MetaCostEvaluator
from .controller import BatchHighwayMPC, EgoState, StepTelemetry

__all__ = [
    "compute_bernstein",
    "compute_bernstein_tensors",
    "BernsteinBasis",
    "BatchTrajectoryOptimizer",
    "OptimizerConfig",
    "OptimizationResult",
    "HighwayTrafficSimulation",
    "ObstacleVehicle",
    "MetaCostEvaluator",
    "BatchHighwayMPC",
    "EgoState",
    "StepTelemetry",
]
