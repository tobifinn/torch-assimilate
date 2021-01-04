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

import numpy as np

# Internal modules
from pytassim.kernels.periodic import PeriodicKernel
from pytassim.kernels.utils import distance_matrix


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestPeriodic(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = PeriodicKernel()
        self.tensor = torch.zeros(10, 2).normal_()

    def test_periodic_kernel_works(self):
        abs_norm = distance_matrix(self.tensor, self.tensor[:3], norm=1)
        inner_product = abs_norm * np.pi / self.kernel.period
        sin_product = 2 * torch.sin(inner_product).pow(2)
        sin_mat = torch.exp(-sin_product / self.kernel.lengthscale ** 2)
        kernel_mat = self.kernel(self.tensor, self.tensor[:3])
        torch.testing.assert_allclose(kernel_mat, sin_mat)

    def test_periodic_kernel_uses_period(self):
        self.kernel.period = 4.
        abs_norm = distance_matrix(self.tensor, self.tensor[:3], norm=1)
        inner_product = abs_norm * np.pi / 4.
        sin_product = 2 * torch.sin(inner_product).pow(2)
        sin_mat = torch.exp(-sin_product / self.kernel.lengthscale ** 2)
        kernel_mat = self.kernel(self.tensor, self.tensor[:3])
        torch.testing.assert_allclose(kernel_mat, sin_mat)

    def test_periodic_kernel_uses_lengthscale(self):
        self.kernel.lengthscale = 4.
        abs_norm = distance_matrix(self.tensor, self.tensor[:3], norm=1)
        inner_product = abs_norm * np.pi / self.kernel.period
        sin_product = 2 * torch.sin(inner_product).pow(2)
        sin_mat = torch.exp(-sin_product / 4. ** 2)
        kernel_mat = self.kernel(self.tensor, self.tensor[:3])
        torch.testing.assert_allclose(kernel_mat, sin_mat)

    def test_kernel_works_for_multidim(self):
        x_tensor = torch.zeros(5, 10, 2).normal_()
        abs_norm = distance_matrix(x_tensor, x_tensor[:, :3], norm=1)
        inner_product = abs_norm * np.pi / self.kernel.period
        sin_product = 2 * torch.sin(inner_product).pow(2)
        sin_mat = torch.exp(-sin_product / self.kernel.lengthscale ** 2)
        kernel_mat = self.kernel(x_tensor, x_tensor[:, :3])
        torch.testing.assert_allclose(kernel_mat, sin_mat)

    def test_kernel_diffbar(self):
        self.kernel.period = torch.nn.Parameter(torch.ones(1))
        self.kernel.lengthscale = torch.nn.Parameter(torch.ones(1))
        kernel_out = self.kernel(self.tensor, self.tensor).mean()

        grad_out = grad(
            [kernel_out], [self.kernel.period, self.kernel.lengthscale],
            retain_graph=True
        )
        kernel_out.backward()
        torch.testing.assert_allclose(self.kernel.period.grad, grad_out[0])
        torch.testing.assert_allclose(self.kernel.lengthscale.grad,
                                      grad_out[1])

    def test_kernel_compilable(self):
        orig_value = self.kernel(self.tensor, self.tensor)
        compiled_kernel = torch.jit.script(self.kernel)
        compiled_value = compiled_kernel(self.tensor, self.tensor)
        torch.testing.assert_allclose(compiled_value, orig_value)

    if __name__ == '__main__':
        unittest.main()
