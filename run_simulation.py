"""
Main Entrypoint Script for Closed-Loop Batch Highway MPC Simulation.
Usage:
    python run_simulation.py --scenario cruise_IDM --steps 50 --save_gif
    python run_simulation.py --scenario HSRL_IDM --steps 60 --save_gif
    python run_simulation.py --scenario RL_IDM --steps 50 --save_gif
"""

import os
import argparse
import time
import copy
from typing import List, Any
import numpy as np
import torch

from mpc_batch.controller import BatchHighwayMPC, StepTelemetry
from mpc_batch.visualizer import HighwayVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Modal Batch MPC Highway Driving Simulation")
    parser.add_argument("--scenario", type=str, default="cruise_IDM",
                        choices=["cruise_IDM", "HSRL_IDM", "RL_IDM"],
                        help="Driving scenario preference")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--steps", type=int, default=40,
                        help="Number of simulation steps to run")
    parser.add_argument("--save_gif", action="store_true",
                        help="Save animated GIF of the scenario")
    parser.add_argument("--save_snapshots", action="store_true", default=True,
                        help="Save snapshot images at intervals")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save generated plots and animations")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="PyTorch device to run on")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Multi-Modal Model Predictive Control through Batch Non-Holonomic Trajectory Optimization")
    print(f"Scenario: {args.scenario} | Steps: {args.steps} | Device: {args.device}")
    print("=" * 80)
    
    mpc = BatchHighwayMPC(scenario=args.scenario, config_path=args.config, device=args.device)
    visualizer = HighwayVisualizer()
    
    telemetry_history: List[StepTelemetry] = []
    vehicles_history: List[List[Any]] = []
    
    start_wall_time = time.time()
    
    for step_idx in range(args.steps):
        # Save snapshot of traffic vehicles before stepping
        current_vehicles = copy.deepcopy(mpc.traffic_sim.vehicles)
        vehicles_history.append(current_vehicles)
        
        # Step MPC
        telem = mpc.step()
        telemetry_history.append(telem)
        
        # Print progress
        status_flag = "COLLISION!" if telem.collision else "OK"
        print(
            f"Step {telem.step:03d} | "
            f"Time: {telem.sim_time:5.2f}s | "
            f"Ego: (x={telem.ego_state.x:6.2f}m, y={telem.ego_state.y:5.2f}m) | "
            f"Speed: {telem.control_v:5.2f} m/s | "
            f"Goal #{telem.best_goal_idx+1:02d}/{mpc.num_goals} | "
            f"Solve Time: {telem.computation_time*1000:5.1f}ms | "
            f"Status: {status_flag}"
        )
        
        # Save snapshots at key intervals
        if args.save_snapshots and (step_idx == 0 or step_idx == args.steps // 2 or step_idx == args.steps - 1):
            snapshot_path = os.path.join(args.output_dir, f"{args.scenario}_step_{step_idx:03d}.png")
            visualizer.save_snapshot(telem, current_vehicles, snapshot_path)
            print(f"  -> Saved snapshot: {snapshot_path}")
            
    total_elapsed = time.time() - start_wall_time
    avg_solve_time = np.mean([t.computation_time for t in telemetry_history]) * 1000.0
    avg_speed = np.mean([t.control_v for t in telemetry_history])
    min_dist_to_right = np.mean([abs(t.ego_state.y - (-10.0)) for t in telemetry_history])
    collision_occurred = any([t.collision for t in telemetry_history])
    
    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY")
    print(f"Scenario:                    {args.scenario}")
    print(f"Total Simulation Time:       {mpc.sim_time:.2f} s")
    print(f"Average Batch Solve Time:    {avg_solve_time:.2f} ms")
    print(f"Average Forward Velocity:    {avg_speed:.2f} m/s")
    print(f"Mean Distance to Right Lane: {min_dist_to_right:.2f} m")
    print(f"Collision Free:              {not collision_occurred}")
    print("=" * 80)
    
    # Save animated GIF if requested
    if args.save_gif:
        gif_path = os.path.join(args.output_dir, f"{args.scenario}_simulation.gif")
        print(f"\nRendering animated GIF to {gif_path} (this may take a few seconds)...")
        visualizer.create_animation(telemetry_history, vehicles_history, gif_path, fps=10)
        print(f"Animation saved: {gif_path}")


if __name__ == "__main__":
    main()

