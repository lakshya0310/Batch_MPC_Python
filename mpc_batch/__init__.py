"""
Batch Non-Holonomic Trajectory Optimization for Multi-Modal Highway Driving MPC.
PyTorch Implementation of the paper:
'Multi-Modal Model Predictive Control through Batch Non-Holonomic Trajectory Optimization: Application to Highway Driving'
Vivek K. Adajania, Aditya Sharma, Anish Gupta, Houman Masnavi, K Madhava Krishna, Arun K. Singh (IEEE RA-L 2022).
"""

from .bernstein import compute_bernstein, BernsteinBasis

__all__ = [
    "compute_bernstein",
    "BernsteinBasis",
]
