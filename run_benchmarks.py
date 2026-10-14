"""
Benchmark Suite for Batch Non-Holonomic Trajectory Optimization.
Reproduces the key empirical findings from the paper:
1. Figure 2(a): Optimizer residual convergence trends across iterations.
2. Figure 2(b): Computation time scaling vs. batch size (L = 2, 4, 6, 8, 10, 11, 12, 14, 16, 20).
3. Tables II, III, IV: Performance comparison between Single-Goal MPC (L=1) and Batch Multi-Modal MPC (L=11).
"""

import os
import time
import argparse
from typing import Dict, List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt

from mpc_batch.optimizer import BatchTrajectoryOptimizer, OptimizerConfig
from mpc_batch.controller import BatchHighwayMPC


def benchmark_residuals(output_dir: str = "outputs"):
    """
    Evaluates and plots constraint residuals across iterations, matching Figure 2(a).
    """
    print("\n" + "=" * 70)
    print("Running Benchmark 1: Constraint Residual Convergence (Figure 2a)")
    print("=" * 70)
    
    config = OptimizerConfig(t_fin=8.0, num=100, maxiter=100)
    opt = BatchTrajectoryOptimizer(config)
    
    # Standard 11 goals in highway
    L = 11
    x_goals = torch.full((L,), 120.0)
    y_goals = torch.linspace(-10.0, 10.0, L)
    
    # 6 typical highway obstacles
    x_obs_init = torch.tensor([25.0, 60.0, 40.0, 35.0, 70.0, 85.0])
    y_obs_init = torch.tensor([-6.0, -2.0, 2.0, 6.0, 10.0, -10.0])
    vx_obs = torch.tensor([11.0, 8.0, 9.5, 8.5, 12.0, 7.5])
    vy_obs = torch.zeros(6)
    
    res = opt.solve(
        x_init=0.0,
        y_init=10.0,
        v_init=14.0,
        psi_init=0.0,
        psidot_init=0.0,
        x_goals=x_goals,
        y_goals=y_goals,
        x_obs_init=x_obs_init,
        y_obs_init=y_obs_init,
        vx_obs=vx_obs,
        vy_obs=vy_obs,
    )
    
    iterations = np.arange(1, config.maxiter + 1)
    res_obs = res.res_obs_history.cpu().numpy()
    res_bounds = res.res_bounds_history.cpu().numpy()
    res_nonhol = res.res_nonhol_history.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, res_obs, label="Collision Avoidance", color="#ff7f0e", linewidth=2.0)
    ax.plot(iterations, res_nonhol, label="Non-holonomic", color="#1f77b4", linewidth=2.0)
    ax.plot(iterations, res_bounds, label="Acceleration Bounds", color="#2ca02c", linewidth=2.0)
    
    ax.set_xlabel("Optimizer Iterations", fontsize=12)
    ax.set_ylabel("Constraint Residuals Norm", fontsize=12)
    ax.set_title("Constraint Residual Trend (Figure 2a)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    
    out_path = os.path.join(output_dir, "benchmark_residuals_fig2a.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Residual plot saved to: {out_path}")
    print(f"Final Residuals: Obs={res_obs[-1]:.4f}, Nonhol={res_nonhol[-1]:.4f}, Bounds={res_bounds[-1]:.4f}")


def benchmark_batch_scaling(output_dir: str = "outputs"):
    """
    Evaluates solve time scaling vs. batch size L, matching Figure 2(b).
    """
    print("\n" + "=" * 70)
    print("Running Benchmark 2: Computation Time vs Batch Size (Figure 2b)")
    print("=" * 70)
    
    config = OptimizerConfig(t_fin=8.0, num=100, maxiter=100)
    opt = BatchTrajectoryOptimizer(config)
    
    batch_sizes = [2, 4, 6, 8, 10, 11, 12, 14, 16, 20]
    solve_times_ms = []
    
    x_obs_init = torch.tensor([25.0, 60.0, 40.0, 35.0, 70.0, 85.0])
    y_obs_init = torch.tensor([-6.0, -2.0, 2.0, 6.0, 10.0, -10.0])
    vx_obs = torch.tensor([11.0, 8.0, 9.5, 8.5, 12.0, 7.5])
    vy_obs = torch.zeros(6)
    
    # Warmup
    _ = opt.solve(0.0, 10.0, 14.0, 0.0, 0.0, torch.full((2,), 120.0), torch.linspace(-2, 2, 2),
                  x_obs_init, y_obs_init, vx_obs, vy_obs)
    
    num_trials = 5
    for L in batch_sizes:
        x_goals = torch.full((L,), 120.0)
        y_goals = torch.linspace(-10.0, 10.0, L)
        
        times = []
        for _ in range(num_trials):
            t0 = time.perf_counter()
            _ = opt.solve(
                x_init=0.0,
                y_init=10.0,
                v_init=14.0,
                psi_init=0.0,
                psidot_init=0.0,
                x_goals=x_goals,
                y_goals=y_goals,
                x_obs_init=x_obs_init,
                y_obs_init=y_obs_init,
                vx_obs=vx_obs,
                vy_obs=vy_obs,
            )
            times.append((time.perf_counter() - t0) * 1000.0)
            
        mean_time = np.mean(times)
        solve_times_ms.append(mean_time)
        print(f"Batch Size L={L:2d} -> Mean Solve Time: {mean_time:6.2f} ms")
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(batch_sizes, solve_times_ms, marker="o", color="#d62728", linewidth=2.5, markersize=7, label="Batch Optimizer (PyTorch)")
    
    ax.set_xlabel("Batch Size (Number of Goals)", fontsize=12)
    ax.set_ylabel("Computation Time (ms)", fontsize=12)
    ax.set_title("Computation Time vs. Batch Size (Figure 2b)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xticks(batch_sizes)
    ax.legend(fontsize=11)
    
    out_path = os.path.join(output_dir, "benchmark_batch_scaling_fig2b.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Batch scaling plot saved to: {out_path}")


def benchmark_single_vs_batch(num_steps: int = 25, output_dir: str = "outputs"):
    """
    Compares Single-Goal MPC (L=1) vs Batch MPC (L=11), reproducing Tables II & IV.
    """
    print("\n" + "=" * 70)
    print("Running Benchmark 3: Single-Goal MPC vs Batch Multi-Modal MPC (Tables II & IV)")
    print("=" * 70)
    
    # 1. Run Single-Goal MPC (L=1, fixed to current lane)
    import yaml
    with open("config.yaml", "r") as f:
        cfg_single = yaml.safe_load(f)
    cfg_single["scenarios"]["cruise_single"] = {
        "name": "Cruise Single Goal",
        "num_goals": 1,
        "x_g": [120.0],
        "y_g": [10.0], # Fixed in lane
        "weights": [50.0, 50.0, 0.0, 0.0],
        "ego_init": {"x": 0.0, "y": 10.0, "v": 14.0, "psi": 0.0},
    }
    
    mpc_single = BatchHighwayMPC(scenario="cruise_single", custom_config=cfg_single)
    mpc_batch = BatchHighwayMPC(scenario="cruise_IDM", config_path="config.yaml")
    
    v_cruise = 15.0
    single_res_v = []
    batch_res_v = []
    single_speeds = []
    batch_speeds = []
    
    print(f"Simulating {num_steps} steps for Single-Goal MPC...")
    for _ in range(num_steps):
        telem = mpc_single.step()
        err_v = (telem.control_v - v_cruise) ** 2
        single_res_v.append(err_v)
        single_speeds.append(telem.control_v)
        
    print(f"Simulating {num_steps} steps for Batch MPC (L=11)...")
    for _ in range(num_steps):
        telem = mpc_batch.step()
        err_v = (telem.control_v - v_cruise) ** 2
        batch_res_v.append(err_v)
        batch_speeds.append(telem.control_v)
        
    print("\n" + "-" * 70)
    print(f"{'Method':<20} | {'Mean (v-vcruise)^2':<20} | {'Max (v-vcruise)^2':<20} | {'Mean Velocity':<15}")
    print("-" * 70)
    print(f"{'Single MPC (L=1)':<20} | {np.mean(single_res_v):<20.4f} | {np.max(single_res_v):<20.4f} | {np.mean(single_speeds):<15.2f}")
    print(f"{'Batch MPC (L=11)':<20} | {np.mean(batch_res_v):<20.4f} | {np.max(batch_res_v):<20.4f} | {np.mean(batch_speeds):<15.2f}")
    print("-" * 70)
    
    # Plot velocity comparison over time
    time_arr = np.arange(1, num_steps + 1) * 0.08
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time_arr, single_speeds, label="Single-Goal MPC (L=1)", linestyle="--", color="#d62728", linewidth=2.0)
    ax.plot(time_arr, batch_speeds, label="Batch MPC (L=11, Ours)", color="#1f77b4", linewidth=2.5)
    ax.axhline(v_cruise, color="black", linestyle=":", label="Target Cruise Velocity (15 m/s)")
    
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Forward Velocity (m/s)", fontsize=12)
    ax.set_title("Velocity Tracking Comparison: Single vs Batch MPC", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    
    out_path = os.path.join(output_dir, "benchmark_single_vs_batch.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Comparison plot saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark suite")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    benchmark_residuals(args.output_dir)
    benchmark_batch_scaling(args.output_dir)
    benchmark_single_vs_batch(num_steps=25, output_dir=args.output_dir)
    print("\nAll benchmarks completed successfully!")


if __name__ == "__main__":
    main()

