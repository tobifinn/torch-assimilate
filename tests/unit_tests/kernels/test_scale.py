#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 19.08.20

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
from pytassim.kernels.scale import ScaleKernel


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestScaleKernel(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = ScaleKernel(scaling=2.)
        self.test_tensor = torch.zeros(10, 2).normal_()

    def test_scale_kernel_returns_valid_kernel_matrix(self):
        ret_k = self.kernel(self.test_tensor, self.test_tensor[4:8])
        self.assertIsInstance(ret_k, torch.Tensor)
        self.assertTupleEqual(ret_k.shape, (10, 4))

    def test_scale_kernel_uses_scaling(self):
        ret_k = self.kernel(self.test_tensor, self.test_tensor)
        torch.testing.assert_allclose(ret_k, self.kernel.scaling)
        self.kernel.scaling = 5.23
        ret_k = self.kernel(self.test_tensor, self.test_tensor)
        torch.testing.assert_allclose(ret_k, self.kernel.scaling)

    def test_scale_works_with_tensor_scaling(self):
        self.kernel.scaling = torch.zeros(10, 10).normal_()
        ret_k = self.kernel(self.test_tensor, self.test_tensor)
        torch.testing.assert_allclose(ret_k, self.kernel.scaling)

        self.kernel.scaling = torch.zeros(10).normal_()
        ret_k = self.kernel(self.test_tensor, self.test_tensor)
        right_k = self.kernel.scaling * torch.ones(10, 10)
        torch.testing.assert_allclose(ret_k, right_k)

        self.kernel.scaling = torch.zeros(10, 1).normal_()
        right_k = self.kernel.scaling * torch.ones(10, 2)
        ret_k = self.kernel(self.test_tensor, self.test_tensor[:2])
        torch.testing.assert_allclose(ret_k, right_k)

    def test_works_with_multidim_input(self):
        self.kernel.scaling = 2.
        tensor = torch.zeros(5, 5, 10, 2).normal_()
        right_mat = torch.ones(5, 5, 10, 10) * self.kernel.scaling
        ret_mat = self.kernel(tensor, tensor)
        torch.testing.assert_allclose(ret_mat, right_mat)

    def test_scale_is_differentiable(self):
        self.kernel.scaling = torch.nn.Parameter(torch.zeros(10, 1).normal_())
        self.assertIsNone(self.kernel.scaling.grad)
        ret_k = self.kernel(self.test_tensor, self.test_tensor).mean()
        right_grad = grad([ret_k], [self.kernel.scaling],
                          retain_graph=True)[0]
        ret_k.backward()
        self.assertIsNotNone(self.kernel.scaling.grad)
        torch.testing.assert_allclose(self.kernel.scaling.grad, right_grad)

    def test_kernel_compilable(self):
        self.kernel.scaling = torch.nn.Parameter(torch.zeros(10, 1).normal_())
        orig_value = self.kernel(self.test_tensor, self.test_tensor)
        compiled_kernel = torch.jit.script(self.kernel)
        compiled_value = compiled_kernel(self.test_tensor, self.test_tensor)
        torch.testing.assert_allclose(compiled_value, orig_value)

    if __name__ == '__main__':
        unittest.main()
