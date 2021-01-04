#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 26.08.20

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
import torch.nn
from torch.autograd import grad

# Internal modules
from pytassim.kernels.orn_uhl import OrnsteinUhlenbeckKernel
from pytassim.kernels.utils import distance_matrix


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestOrnUhl(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = OrnsteinUhlenbeckKernel()
        self.tensor = torch.zeros(10, 2).normal_()

    def test_kernel_returns_right_matrix(self):
        abs_norm = distance_matrix(self.tensor, self.tensor[:2], norm=1)
        factor = -abs_norm/self.kernel.lengthscale
        right_mat = factor.exp()
        kernel_mat = self.kernel(self.tensor, self.tensor[:2])
        torch.testing.assert_allclose(kernel_mat, right_mat)

    def test_kernel_works_for_multidim_tensor(self):
        x_tensor = torch.ones(5, 10, 2).normal_()
        abs_norm = distance_matrix(x_tensor, x_tensor[:, :5], norm=1)
        factor = -abs_norm/self.kernel.lengthscale
        right_mat = factor.exp()
        kernel_mat = self.kernel(x_tensor, x_tensor[:, :5])
        torch.testing.assert_allclose(kernel_mat, right_mat)

    def test_kernel_diffbar(self):
        self.kernel.lengthscale = torch.nn.Parameter(torch.ones(1))
        kernel_out = self.kernel(self.tensor, self.tensor).mean()
        right_grad = grad([kernel_out], [self.kernel.lengthscale],
                          retain_graph=True)[0]
        kernel_out.backward()
        torch.testing.assert_allclose(self.kernel.lengthscale.grad, right_grad)

    def test_kernel_compilable(self):
        orig_value = self.kernel(self.tensor, self.tensor)
        compiled_kernel = torch.jit.script(self.kernel)
        compiled_value = compiled_kernel(self.tensor, self.tensor)
        torch.testing.assert_allclose(compiled_value, orig_value)

    if __name__ == '__main__':
        unittest.main()
