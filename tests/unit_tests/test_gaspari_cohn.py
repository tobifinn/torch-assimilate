#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12/7/18

Created for torch-assimilate

    Copyright (C) {2018}

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
import xarray as xr
import numpy as np

# Internal modules
from pytassim.localization.gaspari_cohn import GaspariCohn
from pytassim.testing import dummy_distance


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestGaspariCohn(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        self.loc = GaspariCohn(5, dist_func=dummy_distance)
        self.dist = dummy_distance(10, self.state.grid.values)

    def test_f1_returns_f1_from_gc99(self):
        ret_f1 = self.loc._f1(self.dist)
        f1 = -28 * self.dist**5 / 33
        f1 += 8 * self.dist**4 / 11
        f1 += 20 * self.dist**3 / 11
        f1 -= 80 * self.dist**2 / 33
        f1 += 1
        np.testing.assert_equal(ret_f1, f1)

    def test_f2_returns_f2_from_gc99(self):
        f2 = 20 * self.dist ** 5 / 33
        f2 -= 16 * self.dist ** 4 / 11
        f2 += 100 * self.dist ** 2 / 33
        f2 -= 45 * self.dist / 11
        f2 += 51 / 22
        f2 -= 7 / (44 * self.dist)

        ret_f2 = self.loc._f2(self.dist)
        np.testing.assert_equal(ret_f2, f2)

    def test_f3_returns_f3_from_gc99(self):
        f3 = -4 * self.dist ** 5 / 11
        f3 += 16 * self.dist ** 4 / 11
        f3 -= 10 * self.dist ** 3 / 11
        f3 -= 100 * self.dist ** 2 / 33
        f3 += 5 * self.dist
        f3 -= 61 / 22
        f3 += 115 / (132 * self.dist)

        ret_f3 = self.loc._f3(self.dist)
        np.testing.assert_equal(ret_f3, f3)

    def test_f4_returns_f4_from_gc99(self):
        f4 = 4 * self.dist ** 5 / 33
        f4 -= 8 * self.dist ** 4 / 11
        f4 += 10 * self.dist ** 3 / 11
        f4 += 80 * self.dist ** 2 / 33
        f4 -= 80 * self.dist / 11
        f4 += 64 / 11
        f4 -= 32 / (33 * self.dist)

        ret_f4 = self.loc._f4(self.dist)
        np.testing.assert_equal(ret_f4, f4)


if __name__ == '__main__':
    unittest.main()
