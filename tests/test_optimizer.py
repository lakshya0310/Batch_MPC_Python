"""
Unit test for BatchTrajectoryOptimizer.
Verifies batched solves, boundary conditions satisfaction, and convergence.
"""

import time
import unittest
import torch
import numpy as np

from mpc_batch.optimizer import BatchTrajectoryOptimizer, OptimizerConfig


class TestBatchTrajectoryOptimizer(unittest.TestCase):
    def setUp(self):
        self.config = OptimizerConfig(
            t_fin=8.0,
            num=100,
            maxiter=100,
            num_obs=6,
            v_min=0.0,
            v_max=24.0,
            a_max=4.0,
        )
        self.optimizer = BatchTrajectoryOptimizer(self.config)

    def test_batch_solve(self):
        # 11 goals as in Cruise IDM scenario
        L = 11
        x_goals = torch.full((L,), 120.0)
        y_goals = torch.tensor([-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        
        # Surrounding obstacle positions and velocities
        x_obs_init = torch.tensor([25.0, 60.0, 40.0, 35.0, 70.0, 85.0])
        y_obs_init = torch.tensor([-6.0, -2.0, 2.0, 6.0, 10.0, -10.0])
        vx_obs = torch.tensor([11.0, 8.0, 9.5, 8.5, 12.0, 7.5])
        vy_obs = torch.zeros(6)
        
        x_init = 0.0
        y_init = 10.0
        v_init = 14.0
        psi_init = 0.0
        psidot_init = 0.0
        
        t0 = time.perf_counter()
        res = self.optimizer.solve(
            x_init=x_init,
            y_init=y_init,
            v_init=v_init,
            psi_init=psi_init,
            psidot_init=psidot_init,
            x_goals=x_goals,
            y_goals=y_goals,
            x_obs_init=x_obs_init,
            y_obs_init=y_obs_init,
            vx_obs=vx_obs,
            vy_obs=vy_obs,
        )
        t_elapsed = time.perf_counter() - t0
        print(f"\n[Test] Solved batch of {L} trajectories in {t_elapsed*1000:.2f} ms")
        
        # 1. Output shapes
        self.assertEqual(res.x.shape, (L, 100))
        self.assertEqual(res.y.shape, (L, 100))
        self.assertEqual(res.psi.shape, (L, 100))
        self.assertEqual(res.v.shape, (L, 100))
        
        # 2. Boundary conditions
        self.assertTrue(torch.allclose(res.x[:, 0], torch.full((L,), x_init), atol=1e-3))
        self.assertTrue(torch.allclose(res.y[:, 0], torch.full((L,), y_init), atol=1e-3))
        self.assertTrue(torch.allclose(res.x[:, -1], x_goals, atol=1e-3))
        self.assertTrue(torch.allclose(res.y[:, -1], y_goals, atol=1e-3))
        self.assertTrue(torch.allclose(res.psi[:, 0], torch.full((L,), psi_init), atol=1e-3))
        
        # 3. Velocity bounds satisfaction
        self.assertTrue(torch.all(res.v >= self.config.v_min - 1e-3))
        self.assertTrue(torch.all(res.v <= self.config.v_max + 1e-3))
        
        # 4. Convergence: residuals at final iteration must be substantially lower than initial/peak
        self.assertLess(res.res_obs_history[-1].item(), res.res_obs_history[0].item())
        self.assertLess(res.res_bounds_history[-1].item(), torch.max(res.res_bounds_history).item())
        self.assertLess(res.res_nonhol_history[-1].item(), res.res_nonhol_history[0].item())


if __name__ == "__main__":
    unittest.main()
