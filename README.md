# Multi-Modal Model Predictive Control through Batch Non-Holonomic Trajectory Optimization

A complete, high-performance **Python (PyTorch)** implementation from scratch of the research paper:

> **"Multi-Modal Model Predictive Control through Batch Non-Holonomic Trajectory Optimization: Application to Highway Driving"**  
> *Vivek K. Adajania, Aditya Sharma, Anish Gupta, Houman Masnavi, K Madhava Krishna, Arun K. Singh*  
> IEEE Robotics and Automation Letters (RA-L), Accepted January 2022.  
> [arXiv:2109.10392](https://arxiv.org/abs/2109.10392)

---

## 🚀 Key Theoretical Insights & Innovations

Standard Model Predictive Control (MPC) performs a local search in non-convex optimization, getting easily trapped in local minima (e.g. stuck behind slow vehicles without discovering overtaking or lane-change maneuvers). This paper solves this challenge by formulating a **batch non-holonomic trajectory optimizer** that:
1. **Solves $L$ Goal-Directed Trajectory Optimizations Simultaneously in Parallel**:
   - Multiple candidate goals across highway lanes are sampled simultaneously.
   - Each goal-directed trajectory converges to a distinct locally optimal motion plan (multi-modal behavior).
2. **Linearization-Free Multi-Convex Reformulation**:
   - Reformulates quadratic collision avoidance and acceleration bounds using auxiliary polar variables ($d_{obs}, \alpha_{obs}$ and $d_a, \alpha_a$).
   - Avoids expensive per-iteration Jacobian recalculations or SQP matrix factorizations.
3. **One-Shot Batched Linear Algebra (Alternating Minimization / Split-Bregman)**:
   - The primary computational bottleneck reduces to solving a constant KKT linear system $\mathbf{Q}_{xy} \mathbf{c}_{xy} = \mathbf{q}$.
   - Because the system matrix $\mathbf{Q}_{xy}$ is **identical across all batch problem instances $l=1,\dots,L$**, parallelizing across arbitrary batch sizes boils down to a single batched matrix-vector multiplication $\mathbf{Q}_{xy}^{-1} [\mathbf{q}_1 | \mathbf{q}_2 | \dots | \mathbf{q}_L]$.
   - Scales linearly with batch size, running up to $6\times$ faster than multi-threaded SQP solvers (ACADO).
4. **Multi-Modal Meta-Cost Ranking**:
   - Evaluates resulting trajectories against user-defined high-level driving objectives (Cruise driving, High-Speed Right-Lane preference, Obstacle clearance).

---

## 🛠️ Architecture & 1-to-1 C++ to Python/PyTorch Mapping

| Paper Component | Original C++ Implementation | This Python (PyTorch) Implementation | Description |
|---|---|---|---|
| **Trajectory Basis** | `bernstein_coeff_order10_new` | [`mpc_batch.bernstein`](file:///c:/Users/Hi/MDPCalib_paper/mpc_batch/bernstein.py) | Order-10 Bernstein polynomial basis ($P, \dot{P}, \ddot{P}$) |
| **Batch Optimizer** | `optim::solve`, `compute_x`, `compute_psi` | [`mpc_batch.optimizer.BatchTrajectoryOptimizer`](file:///c:/Users/Hi/MDPCalib_paper/mpc_batch/optimizer.py) | Alternating Minimization with batched PyTorch tensor operations |
| **Traffic Dynamics** | `highway_car2.py:IDM` | [`mpc_batch.traffic.HighwayTrafficSimulation`](file:///c:/Users/Hi/MDPCalib_paper/mpc_batch/traffic.py) | 6-lane highway simulation with Intelligent Driver Model (IDM) |
| **Meta-Cost Ranking** | `MinimalPublisher::get_ranks` | [`mpc_batch.meta_cost.MetaCostEvaluator`](file:///c:/Users/Hi/MDPCalib_paper/mpc_batch/meta_cost.py) | Multi-criteria ordinal ranking and trajectory selection |
| **Closed-Loop MPC** | `MinimalPublisher::timer_callback` | [`mpc_batch.controller.BatchHighwayMPC`](file:///c:/Users/Hi/MDPCalib_paper/mpc_batch/controller.py) | Receding-horizon MPC execution loop |
| **Visualization** | `highway_car2.py:plt` | [`mpc_batch.visualizer.HighwayVisualizer`](file:///c:/Users/Hi/MDPCalib_paper/mpc_batch/visualizer.py) | Highway rendering, ellipses, rectangles, candidate trajectories, and GIFs |

---

## 📁 Repository Structure

```
MDPCalib_paper/
├── .gitignore
├── requirements.txt            # Python dependencies (PyTorch, NumPy, Matplotlib, PyYAML)
├── config.yaml                 # Scenario configurations and optimizer hyperparameters
├── mpc_batch/
│   ├── __init__.py             # Clean package exports
│   ├── bernstein.py            # Order-10 Bernstein polynomial trajectory parameterization
│   ├── optimizer.py            # Batched Non-Holonomic Trajectory Optimizer (PyTorch)
│   ├── traffic.py              # Highway simulation with IDM car-following dynamics
│   ├── meta_cost.py            # Meta-cost evaluation and multi-modal trajectory ranking
│   ├── controller.py           # Closed-loop receding-horizon Model Predictive Control
│   └── visualizer.py           # High-resolution matplotlib highway visualizer and animator
├── tests/
│   ├── test_bernstein.py       # Unit test verifying Bernstein basis, unity, and derivatives
│   ├── test_optimizer.py       # Unit test verifying batch convergence and boundary conditions
│   ├── test_traffic.py         # Unit test verifying IDM dynamics and collision checking
│   └── test_controller.py      # Unit test verifying closed-loop MPC execution
├── run_simulation.py           # Interactive simulation runner
├── run_benchmarks.py           # Reproduces paper experiments (Figures 2a, 2b, Tables II & IV)
└── README.md
```

---

## ⚡ Quick Start

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Full Test Suite

All unit tests verify mathematical equivalence, convergence, and execution integrity:

```bash
python -m unittest discover tests
```

### 3. Run Closed-Loop Highway MPC Simulation

Run a multi-modal highway simulation in any driving scenario:

```bash
# Scenario 1: Cruise Driving with IDM Traffic (11 goal candidates)
python run_simulation.py --scenario cruise_IDM --steps 40 --save_snapshots

# Scenario 2: High-Speed Driving with Right-Lane Preference (HSRL)
python run_simulation.py --scenario HSRL_IDM --steps 50 --save_snapshots

# Scenario 3: Right-Lane Preference (RL)
python run_simulation.py --scenario RL_IDM --steps 40 --save_snapshots

# Optional: Generate full animated GIF
python run_simulation.py --scenario cruise_IDM --steps 30 --save_gif
```

Snapshots and animations are saved to `outputs/`.

---

## 📊 Benchmark Suite (Reproducing Paper Results)

Run the comprehensive benchmark suite:

```bash
python run_benchmarks.py
```

This reproduces:
1. **Figure 2(a) - Constraint Residual Trends**: Verifies that collision avoidance ($\|res_{obs}\|_2$), non-holonomic kinematics ($\|res_{nonhol}\|_2$), and acceleration bounds ($\|res_{bounds}\|_2$) converge under the Split-Bregman parameter ramping.
2. **Figure 2(b) - Batch Scaling vs. Computation Time**: Evaluates solve time across batch sizes $L \in [2, 4, 6, 8, 10, 11, 12, 14, 16, 20]$. Demonstrates near-linear scalability enabled by one-shot batched linear algebra.
3. **Tables II & IV - Single-Goal MPC vs. Batch Multi-Modal MPC**: Demonstrates that standard single-goal MPC gets trapped behind slow obstacles, whereas the batch MPC discovers lane-change and overtaking maneuvers to maintain target speed and right-lane preference.

Generated plots are placed in `outputs/`:
- `benchmark_residuals_fig2a.png`
- `benchmark_batch_scaling_fig2b.png`
- `benchmark_single_vs_batch.png`

---

## 📖 Citation

```bibtex
@article{adajania2022multi,
  title={Multi-Modal Model Predictive Control through Batch Non-Holonomic Trajectory Optimization: Application to Highway Driving},
  author={Adajania, Vivek K and Sharma, Aditya and Gupta, Anish and Masnavi, Houman and Krishna, K Madhava and Singh, Arun K},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={2},
  pages={3320--3327},
  year={2022},
  publisher={IEEE}
}
```

