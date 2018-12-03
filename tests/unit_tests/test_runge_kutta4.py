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
import torch


# Internal modules
from pytassim.model.integration.rk4 import RK4Integrator


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


def dummy_model(state):
    return state * 2


class TestRK4Integrator(unittest.TestCase):
    def setUp(self):
        self.integrator = RK4Integrator(model=dummy_model)
        self.state = np.array([1., ])

    def test_estimate_slopes_returns_averaged_slope(self):
        k1 = dummy_model(self.state)
        k2 = dummy_model(self.state + k1 * self.integrator.dt / 2)
        k3 = dummy_model(self.state + k2 * self.integrator.dt / 2)
        k4 = dummy_model(self.state + k3 * self.integrator.dt)

        averaged_slope = (k1 + 2 * k2 + 2*k3 + k4) / 6

        returned_slope = self.integrator._estimate_slope(self.state)
        np.testing.assert_equal(returned_slope, averaged_slope)

    def test_calc_increment_is_slope_dt(self):
        returned_slope = self.integrator._estimate_slope(self.state)
        right_increment = returned_slope * self.integrator.dt
        returned_increment = self.integrator._calc_increment(self.state)
        np.testing.assert_equal(returned_increment, right_increment)

    def test_rk4_can_be_used_with_pytorch(self):
        torch_state = torch.ones((1, ))
        torch_increment = self.integrator._calc_increment(torch_state)
        right_increment = self.integrator._calc_increment(self.state)
        np.testing.assert_almost_equal(torch_increment.numpy(), right_increment)


if __name__ == '__main__':
    unittest.main()
