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

# Internal modules
from pytassim.kernels.linear import LinearKernel
from pytassim.kernels.utils import dot_product


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestLinearKernel(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = LinearKernel()
        self.tensor = torch.zeros(10, 2).normal_()

    def test_linear_kernel_makes_dot_product(self):
        ret_k = self.kernel(self.tensor, self.tensor)
        right_k = torch.mm(self.tensor, self.tensor.t())
        torch.testing.assert_allclose(ret_k, right_k)
        ret_k = self.kernel(self.tensor, self.tensor[:2])
        right_k = torch.mm(self.tensor, self.tensor[:2].t())
        torch.testing.assert_allclose(ret_k, right_k)

    def test_dot_product_makes_dot_product(self):
        ret_k = dot_product(self.tensor, self.tensor)
        right_k = torch.mm(self.tensor, self.tensor.t())
        torch.testing.assert_allclose(ret_k, right_k)
        ret_k = dot_product(self.tensor, self.tensor[:2])
        right_k = torch.mm(self.tensor, self.tensor[:2].t())
        torch.testing.assert_allclose(ret_k, right_k)

    def test_linear_kernel_uses_dot_product(self):
        ret_k = dot_product(self.tensor, self.tensor)
        trg = 'pytassim.kernels.utils.dot_product'
        with patch(trg, return_value=ret_k) as dot_patch:
            _ = self.kernel(self.tensor, self.tensor)
        dot_patch.assert_called_once_with(self.tensor, self.tensor)


if __name__ == '__main__':
    unittest.main()
