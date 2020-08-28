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

# Internal modules
from pytassim.kernels import base_kernels
from pytassim.kernels.scale import ScaleKernel


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestBaseKernel(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = base_kernels.BaseKernel()
        self.other_kernel = base_kernels.BaseKernel()

    def test_multiply_returns_multiplicative_kernel(self):
        new_kernel = self.kernel * self.other_kernel
        self.assertIsInstance(new_kernel, base_kernels.MultiplicativeKernel)
        self.assertEqual(id(new_kernel.kernel_1), id(self.kernel))
        self.assertEqual(id(new_kernel.kernel_2), id(self.other_kernel))

    def test_addition_returns_additive_kernel(self):
        new_kernel = self.kernel + self.other_kernel
        self.assertIsInstance(new_kernel, base_kernels.AdditiveKernel)
        self.assertEqual(id(new_kernel.kernel_1), id(self.kernel))
        self.assertEqual(id(new_kernel.kernel_2), id(self.other_kernel))

    def test_power_returns_power_kernel(self):
        new_kernel = self.kernel ** self.other_kernel
        self.assertIsInstance(new_kernel, base_kernels.PowerKernel)
        self.assertEqual(id(new_kernel.kernel_1), id(self.kernel))
        self.assertEqual(id(new_kernel.kernel_2), id(self.other_kernel))


class TestCompKernel(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = ScaleKernel(torch.zeros(10, 10).normal_())
        self.other_kernel = ScaleKernel(torch.zeros(10, 10).normal_())
        self.tensor = torch.zeros(10, 1).normal_()

    def test_comp_kernel_sets_modules(self):
        comp_kernel = base_kernels.CompKernel(self.kernel, self.other_kernel)
        childs = list(comp_kernel.children())
        self.assertEqual(id(childs[0]), id(self.kernel))
        self.assertEqual(id(childs[1]), id(self.other_kernel))
        self.assertEqual(id(comp_kernel.kernel_1), id(self.kernel))
        self.assertEqual(id(comp_kernel.kernel_2), id(self.other_kernel))

    def test_multiplicative_kernel_multiplies_kernels(self):
        mul_kernel = base_kernels.MultiplicativeKernel(
            self.kernel, self.other_kernel
        )
        ret_k = mul_kernel(self.tensor, self.tensor)
        right_k = self.kernel.scaling * self.other_kernel.scaling
        torch.testing.assert_allclose(ret_k, right_k)

    def test_additive_kernel_adds_kernels(self):
        add_kernel = base_kernels.AdditiveKernel(
            self.kernel, self.other_kernel
        )
        ret_k = add_kernel(self.tensor, self.tensor)
        right_k = self.kernel.scaling + self.other_kernel.scaling
        torch.testing.assert_allclose(ret_k, right_k)

    def test_power_kernel_exponents_kernels(self):
        self.other_kernel.scaling = torch.zeros(10, 10).uniform_(1, 2)
        pow_kernel = base_kernels.PowerKernel(
            self.kernel, self.other_kernel
        )
        ret_k = pow_kernel(self.tensor, self.tensor)
        right_k = torch.pow(self.kernel.scaling,  self.other_kernel.scaling)
        torch.testing.assert_allclose(ret_k, right_k)


if __name__ == '__main__':
    unittest.main()
