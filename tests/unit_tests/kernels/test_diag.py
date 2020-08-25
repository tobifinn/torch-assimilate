#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 18.08.20

Created for torch-assimilate

@author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de

    Copyright (C) {2020}  {Tobias Sebastian Finn}

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# System modules
import unittest
import logging
import os

# External modules
import torch
from torch.autograd import grad

# Internal modules
from pytassim.kernels.diag import DiagKernel


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestDiagKernel(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = DiagKernel(scaling=2.)
        self.test_tensor = torch.zeros(10, 2).normal_()

    def test_returns_zeros_if_unequal_dimensions(self):
        ret_mat = self.kernel(self.test_tensor, self.test_tensor[[0]])
        right_mat = torch.zeros(10, 1)
        torch.testing.assert_allclose(ret_mat, right_mat)

    def test_returns_diag_matrix_if_equal_dimensions(self):
        ret_mat = self.kernel(self.test_tensor, self.test_tensor)
        unscaled_mat = ret_mat / self.kernel.scaling
        torch.testing.assert_allclose(unscaled_mat, torch.eye(10))
        torch.testing.assert_allclose(torch.diag(ret_mat), self.kernel.scaling)

    def test_works_with_tensor_scaling(self):
        self.kernel.scaling = torch.zeros(10).normal_()
        ret_mat = self.kernel(self.test_tensor, self.test_tensor)
        torch.testing.assert_allclose(torch.diag(ret_mat), self.kernel.scaling)

    def test_is_differentiable(self):
        self.kernel.scaling = torch.nn.Parameter(torch.zeros(10).normal_())
        self.assertIsNone(self.kernel.scaling.grad)
        ret_mat = self.kernel(self.test_tensor, self.test_tensor)
        ret_mat.mean().backward(retain_graph=True)
        self.assertIsNotNone(self.kernel.scaling.grad)
        self.assertIsInstance(self.kernel.scaling.grad, torch.FloatTensor)
        eval_grad = grad([ret_mat.mean()], [self.kernel.scaling])[0]
        torch.testing.assert_allclose(self.kernel.scaling.grad, eval_grad)


if __name__ == '__main__':
    unittest.main()
