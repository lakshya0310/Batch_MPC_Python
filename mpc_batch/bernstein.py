"""
Bernstein Polynomial Trajectory Parameterization (Order 10)
Directly mirrors the authors' C++ implementation `bernstein_coeff_order10_new`
in `mpc_car_batch/src/optim_batch.cpp`.
"""

import math
from typing import Tuple, NamedTuple, Union
import torch
import numpy as np


class BernsteinBasis(NamedTuple):
    P: torch.Tensor       # [num, 11] Position basis matrix
    Pdot: torch.Tensor    # [num, 11] Velocity basis matrix (d/dt)
    Pddot: torch.Tensor   # [num, 11] Acceleration basis matrix (d^2/dt^2)


def binomial_coeff(n: int, k: int) -> float:
    """Computes binomial coefficient C(n, k)."""
    return float(math.comb(n, k))


def compute_bernstein_tensors(
    tot_time: torch.Tensor,
    t_fin: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> BernsteinBasis:
    """
    Computes Order-10 Bernstein polynomial basis matrices P, Pdot, Pddot.
    Matches the exact polynomial expressions in `optim_batch.cpp:bernstein_coeff_order10_new`.
    
    Args:
        tot_time: 1D Tensor of timestamps in [0, t_fin] of shape [num]
        t_fin: Final time horizon (scalar)
        device: Torch device (CPU or CUDA)
        dtype: Data type (default float32)
        
    Returns:
        BernsteinBasis namedtuple containing (P, Pdot, Pddot)
    """
    tot_time = tot_time.to(device=device, dtype=dtype)
    num = tot_time.shape[0]
    tmin = float(tot_time[0].item())
    tmax = float(t_fin)
    l = tmax - tmin
    
    # Normalized time in [0, 1]
    t = (tot_time - tmin) / l
    one_minus_t = 1.0 - t
    
    n = 10
    nvar = n + 1
    
    P = torch.zeros((num, nvar), device=device, dtype=dtype)
    Pdot_raw = torch.zeros((num, nvar), device=device, dtype=dtype)
    Pddot_raw = torch.zeros((num, nvar), device=device, dtype=dtype)
    
    # Position basis functions P_k(t) = C(10, k) * (1 - t)^(10-k) * t^k
    for k in range(nvar):
        c = binomial_coeff(n, k)
        term1 = torch.pow(one_minus_t, n - k)
        term2 = torch.pow(t, k)
        P[:, k] = c * term1 * term2
        
    # Velocity basis derivatives dP/d(t_norm) as defined in optim_batch.cpp
    Pdot_raw[:, 0] = -10.0 * torch.pow(one_minus_t, 9)
    Pdot_raw[:, 1] = -90.0 * t * torch.pow(one_minus_t, 8) + 10.0 * torch.pow(one_minus_t, 9)
    Pdot_raw[:, 2] = -360.0 * torch.pow(t, 2) * torch.pow(one_minus_t, 7) + 90.0 * t * torch.pow(one_minus_t, 8)
    Pdot_raw[:, 3] = -840.0 * torch.pow(t, 3) * torch.pow(one_minus_t, 6) + 360.0 * torch.pow(t, 2) * torch.pow(one_minus_t, 7)
    Pdot_raw[:, 4] = -1260.0 * torch.pow(t, 4) * torch.pow(one_minus_t, 5) + 840.0 * torch.pow(t, 3) * torch.pow(one_minus_t, 6)
    Pdot_raw[:, 5] = -1260.0 * torch.pow(t, 5) * torch.pow(one_minus_t, 4) + 1260.0 * torch.pow(t, 4) * torch.pow(one_minus_t, 5)
    Pdot_raw[:, 6] = -840.0 * torch.pow(t, 6) * torch.pow(one_minus_t, 3) + 1260.0 * torch.pow(t, 5) * torch.pow(one_minus_t, 4)
    Pdot_raw[:, 7] = -360.0 * torch.pow(t, 7) * torch.pow(one_minus_t, 2) + 840.0 * torch.pow(t, 6) * torch.pow(one_minus_t, 3)
    Pdot_raw[:, 8] = 45.0 * torch.pow(t, 8) * (2.0 * t - 2.0) + 360.0 * torch.pow(t, 7) * torch.pow(one_minus_t, 2)
    Pdot_raw[:, 9] = -10.0 * torch.pow(t, 9) + 9.0 * torch.pow(t, 8) * (-10.0 * t + 10.0)
    Pdot_raw[:, 10] = 10.0 * torch.pow(t, 9)
    
    # Acceleration basis derivatives d^2P/d(t_norm)^2 as defined in optim_batch.cpp
    Pddot_raw[:, 0] = 90.0 * torch.pow(one_minus_t, 8)
    Pddot_raw[:, 1] = 720.0 * t * torch.pow(one_minus_t, 7) - 180.0 * torch.pow(one_minus_t, 8)
    Pddot_raw[:, 2] = 2520.0 * torch.pow(t, 2) * torch.pow(one_minus_t, 6) - 1440.0 * t * torch.pow(one_minus_t, 7) + 90.0 * torch.pow(one_minus_t, 8)
    Pddot_raw[:, 3] = 5040.0 * torch.pow(t, 3) * torch.pow(one_minus_t, 5) - 5040.0 * torch.pow(t, 2) * torch.pow(one_minus_t, 6) + 720.0 * t * torch.pow(one_minus_t, 7)
    Pddot_raw[:, 4] = 6300.0 * torch.pow(t, 4) * torch.pow(one_minus_t, 4) - 10080.0 * torch.pow(t, 3) * torch.pow(one_minus_t, 5) + 2520.0 * torch.pow(t, 2) * torch.pow(one_minus_t, 6)
    Pddot_raw[:, 5] = 5040.0 * torch.pow(t, 5) * torch.pow(one_minus_t, 3) - 12600.0 * torch.pow(t, 4) * torch.pow(one_minus_t, 4) + 5040.0 * torch.pow(t, 3) * torch.pow(one_minus_t, 5)
    Pddot_raw[:, 6] = 2520.0 * torch.pow(t, 6) * torch.pow(one_minus_t, 2) - 10080.0 * torch.pow(t, 5) * torch.pow(one_minus_t, 3) + 6300.0 * torch.pow(t, 4) * torch.pow(one_minus_t, 4)
    Pddot_raw[:, 7] = -360.0 * torch.pow(t, 7) * (2.0 * t - 2.0) - 5040.0 * torch.pow(t, 6) * torch.pow(one_minus_t, 2) + 5040.0 * torch.pow(t, 5) * torch.pow(one_minus_t, 3)
    Pddot_raw[:, 8] = 90.0 * torch.pow(t, 8) + 720.0 * torch.pow(t, 7) * (2.0 * t - 2.0) + 2520.0 * torch.pow(t, 6) * torch.pow(one_minus_t, 2)
    Pddot_raw[:, 9] = -180.0 * torch.pow(t, 8) + 72.0 * torch.pow(t, 7) * (-10.0 * t + 10.0)
    Pddot_raw[:, 10] = 90.0 * torch.pow(t, 8)
    
    # Scale derivatives by chain rule: d/dt_actual = (1/l)*d/dt, d^2/dt_actual^2 = (1/l^2)*d^2/dt^2
    Pdot = Pdot_raw / l
    Pddot = Pddot_raw / (l * l)
    
    return BernsteinBasis(P=P, Pdot=Pdot, Pddot=Pddot)


def compute_bernstein(
    t_fin: float = 8.0,
    num: int = 100,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> BernsteinBasis:
    """
    Convenience function matching `optim::compute_bernstein` in C++.
    Discretizes [0, t_fin] into `num` equidistant points and generates P, Pdot, Pddot.
    """
    tot_time = torch.linspace(0.0, t_fin, steps=num, device=device, dtype=dtype)
    return compute_bernstein_tensors(tot_time, t_fin, device=device, dtype=dtype)

