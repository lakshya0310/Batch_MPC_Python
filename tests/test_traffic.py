"""
Unit test for HighwayTrafficSimulation and IDM car-following dynamics.
"""

import unittest
import numpy as np

from mpc_batch.traffic import HighwayTrafficSimulation


class TestHighwayTrafficSimulation(unittest.TestCase):
    def setUp(self):
        self.sim = HighwayTrafficSimulation(scenario="cruise_IDM", dt=0.08)

    def test_initialization(self):
        self.assertEqual(len(self.sim.vehicles), 18)
        lanes = set([v.y for v in self.sim.vehicles])
        self.assertEqual(len(lanes), 6)

    def test_idm_step(self):
        initial_x = [v.x for v in self.sim.vehicles]
        self.sim.step_idm(ego_x=0.0, ego_y=10.0, ego_v=14.0)
        updated_x = [v.x for v in self.sim.vehicles]
        
        # Vehicles with positive velocity should have moved forward
        for x0, x1 in zip(initial_x, updated_x):
            self.assertGreaterEqual(x1, x0)

    def test_tracked_obstacles_selection(self):
        x_obs, y_obs, vx_obs, vy_obs = self.sim.get_tracked_obstacles(ego_x=0.0, ego_y=10.0)
        self.assertEqual(len(x_obs), 6)
        self.assertEqual(len(y_obs), 6)
        self.assertEqual(len(vx_obs), 6)
        self.assertEqual(len(vy_obs), 6)
        
        # Selected obstacles should not be far behind ego
        for x in x_obs:
            self.assertGreater(x, -25.0)

    def test_collision_detection(self):
        # Place ego right on top of obstacle 0
        veh0 = self.sim.vehicles[0]
        self.assertTrue(self.sim.check_collision(veh0.x, veh0.y))
        
        # Place ego in clear space
        self.assertFalse(self.sim.check_collision(1000.0, 1000.0))


if __name__ == "__main__":
    unittest.main()
