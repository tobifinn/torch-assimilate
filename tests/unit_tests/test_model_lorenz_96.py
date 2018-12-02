#!/bin/env python
# -*- coding: utf-8 -*-
#
#Created on 02.12.18
#
#Created for torch-assim
#
#@author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2018}  {Tobias Sebastian Finn}
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# System modules
import os
import unittest
import logging

# External modules
import numpy as np
import torch

# Internal modules
from pytassim.model.lorenz_96 import torch_roll


rnd = np.random.RandomState(42)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

logging.basicConfig(level=logging.DEBUG)


class TestLorenz96(unittest.TestCase):
    def test_torch_roll(self):
        in_array = rnd.normal(size=100)
        rolled_array = np.roll(in_array, shift=4, axis=0)
        torch_array = torch.tensor(in_array)
        returned_array = torch_roll(torch_array, 4).numpy().copy()
        np.testing.assert_equal(returned_array, rolled_array)
        rolled_array = np.roll(in_array, shift=-4, axis=0)
        returned_array = torch_roll(torch_array, shift=-4).numpy().copy()
        np.testing.assert_equal(returned_array, rolled_array)


if __name__ == '__main__':
    unittest.main()
