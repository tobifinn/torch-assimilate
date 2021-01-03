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
from unittest.mock import patch

# External modules
import torch
import torch.nn
from torch.autograd import grad

# Internal modules
from pytassim.kernels.module_kernel import ModuleKernel


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestModuleKernel(unittest.TestCase):
    def setUp(self) -> None:
        self.small_nn = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 8)
        )
        self.kernel = ModuleKernel(self.small_nn)
        self.tensor = torch.zeros(10, 2).normal_()

    def test_sets_net(self):
        childs = dict(self.kernel.named_children())
        self.assertIn('transform', childs.keys())
        self.assertEqual(childs['transform'], self.small_nn)

    def test_uses_nn(self):
        nn_tensor = self.small_nn(self.tensor)
        nn_sliced = nn_tensor[:2]
        dot_product = nn_tensor @ nn_sliced.t()
        ret_k = self.kernel(self.tensor, self.tensor[:2])
        torch.testing.assert_allclose(ret_k, dot_product)

    @patch('pytassim.kernels.module_kernel.dot_product', return_value=None)
    def test_uses_dot_product(self, dot_patch):
        _ = self.kernel(self.tensor, self.tensor)
        dot_patch.assert_called_once()

    def test_kernel_differentiable(self):
        self.kernel.transform.requires_grad_(True)
        self.assertIsNone(self.kernel.transform[0].weight.grad)
        k_mat = self.kernel(self.tensor, self.tensor[:5]).mean()
        right_grad = grad(k_mat, self.kernel.transform[0].weight,
                          retain_graph=True)[0]
        k_mat.backward()
        torch.testing.assert_allclose(self.kernel.transform[0].weight.grad,
                                      right_grad)

    def test_kernel_compilable(self):
        orig_value = self.kernel(self.tensor, self.tensor)
        compiled_kernel = torch.jit.script(self.kernel)
        compiled_value = compiled_kernel(self.tensor, self.tensor)
        torch.testing.assert_allclose(compiled_value, orig_value)

    if __name__ == '__main__':
        unittest.main()
