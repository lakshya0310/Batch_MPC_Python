"""
Unit test for Order-10 Bernstein polynomial trajectory basis.
Verifies partition of unity, boundary conditions, and derivative consistency.
"""

import unittest
import torch
import numpy as np
from mpc_batch.bernstein import compute_bernstein, BernsteinBasis


class TestBernsteinBasis(unittest.TestCase):
    def setUp(self):
        self.t_fin = 8.0
        self.num = 100
        self.basis = compute_bernstein(t_fin=self.t_fin, num=self.num)

    def test_shapes(self):
        self.assertEqual(self.basis.P.shape, (self.num, 11))
        self.assertEqual(self.basis.Pdot.shape, (self.num, 11))
        self.assertEqual(self.basis.Pddot.shape, (self.num, 11))

    def test_partition_of_unity(self):
        """Sum of Bernstein basis functions at any t must equal 1.0"""
        row_sums = torch.sum(self.basis.P, dim=1)
        expected = torch.ones(self.num)
        self.assertTrue(torch.allclose(row_sums, expected, atol=1e-5))

    def test_boundary_conditions(self):
        """At t=0, only basis 0 is 1. At t=t_fin, only basis 10 is 1."""
        first_row = self.basis.P[0]
        last_row = self.basis.P[-1]
        
        self.assertAlmostEqual(first_row[0].item(), 1.0, places=5)
        self.assertTrue(torch.allclose(first_row[1:], torch.zeros(10), atol=1e-5))
        
        self.assertAlmostEqual(last_row[-1].item(), 1.0, places=5)
        self.assertTrue(torch.allclose(last_row[:-1], torch.zeros(10), atol=1e-5))

    def test_numerical_derivative_consistency(self):
        """Verify analytical Pdot and Pddot with central finite differences."""
        eps = 1e-4
        t_mid = 4.0
        
        t_points = torch.tensor([t_mid - eps, t_mid, t_mid + eps], dtype=torch.float32)
        from mpc_batch.bernstein import compute_bernstein_tensors
        eval_basis = compute_bernstein_tensors(t_points, self.t_fin)
        
        fd_dot = (eval_basis.P[2] - eval_basis.P[0]) / (2.0 * eps)
        analytical_dot = eval_basis.Pdot[1]
        self.assertTrue(torch.allclose(fd_dot, analytical_dot, atol=1e-3))

        fd_ddot = (eval_basis.Pdot[2] - eval_basis.Pdot[0]) / (2.0 * eps)
        analytical_ddot = eval_basis.Pddot[1]
        self.assertTrue(torch.allclose(fd_ddot, analytical_ddot, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
