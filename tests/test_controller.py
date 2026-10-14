"""
Unit test for BatchHighwayMPC closed-loop controller.
"""

import unittest
import numpy as np
import torch

from mpc_batch.controller import BatchHighwayMPC


class TestBatchHighwayMPC(unittest.TestCase):
    def setUp(self):
        self.mpc = BatchHighwayMPC(scenario="cruise_IDM", config_path="config.yaml")

    def test_controller_step(self):
        # Step through 5 MPC iterations
        for i in range(5):
            telem = self.mpc.step()
            self.assertEqual(telem.step, i + 1)
            self.assertGreater(telem.ego_state.x, -1.0)
            self.assertGreater(telem.control_v, 0.0)
            self.assertLessEqual(telem.control_v, 24.0)
            self.assertFalse(telem.collision)
            self.assertIn("best_idx", telem.meta_metrics)
            print(f"Step {telem.step}: x={telem.ego_state.x:.2f}, y={telem.ego_state.y:.2f}, "
                  f"v={telem.control_v:.2f} m/s, goal_idx={telem.best_goal_idx}, "
                  f"comp_time={telem.computation_time*1000:.1f} ms")


if __name__ == "__main__":
    unittest.main()
