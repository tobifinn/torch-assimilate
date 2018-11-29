#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11/29/18

Created for torch-assimilate

@author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de

    Copyright (C) {2018}  {Tobias Sebastian Finn}

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
import numpy as np

# Internal modules
from tfassim.model.integration.rk4 import RK4Integrator


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


def dummy_model(state):
    return state * 2


class TestRK4Integrator(unittest.TestCase):
    def setUp(self):
        self.integrator = RK4Integrator(model=dummy_model)
        self.state = np.array([1, ])

    def test_estimate_slopes_returns_averaged_slope(self):
        right_slopes = [2, 2.2, 2.22, 2.444]
        averaged_slope = np.array(right_slopes).dot(self.integrator._weights)
        slope = self.integrator._estimate_slope(self.state)
        np.testing.assert_equal(slope, averaged_slope)

    def test_calc_increment_is_slope_dt(self):
        returned_slope = self.integrator._estimate_slope(self.state)
        right_increment = returned_slope * self.integrator.dt
        returned_increment = self.integrator._calc_increment(self.state)
        np.testing.assert_equal(returned_increment, right_increment)


if __name__ == '__main__':
    unittest.main()
