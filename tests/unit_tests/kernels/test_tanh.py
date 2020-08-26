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
from unittest.mock import patch

# External modules
import torch
import torch.nn
from torch.autograd import grad

# Internal modules
from pytassim.kernels.tanh import TanhKernel
from pytassim.kernels.utils import dot_product


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestTanhKernel(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = TanhKernel()
        self.tensor = torch.zeros(10, 2).normal_()

    def test_forward_uses_dot_product(self):
        y_tensor = self.tensor[:2]
        dot_prod_out = dot_product(self.tensor, y_tensor)
        trg = 'pytassim.kernels.tanh.dot_product'
        with patch(trg, return_value=dot_prod_out) as dot_patch:
            _ = self.kernel(self.tensor, y_tensor)
        dot_patch.assert_called_once_with(self.tensor, y_tensor)

    def test_forward_returns_tanh_of_logit(self):
        y_tensor = self.tensor[:2]
        dot_prod_out = dot_product(self.tensor, y_tensor)
        logit = self.kernel.coeff * dot_prod_out + self.kernel.const
        right_out = torch.tanh(logit)
        kernel_out = self.kernel(self.tensor, y_tensor)
        torch.testing.assert_allclose(kernel_out, right_out)

    def test_forward_works_with_multidim_input(self):
        x_tensor = torch.zeros(5, 10, 2).normal_()
        y_tensor = torch.zeros(5, 1, 2).normal_()
        dot_prod_out = torch.einsum('ijk,ilk->ijl', x_tensor, y_tensor)
        logit = self.kernel.coeff * dot_prod_out + self.kernel.const
        right_out = torch.tanh(logit)
        kernel_out = self.kernel(x_tensor, y_tensor)
        torch.testing.assert_allclose(kernel_out, right_out)


if __name__ == '__main__':
    unittest.main()
