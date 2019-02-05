#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 1/30/19

Created for torch-assimilate

@author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de

    Copyright (C) {2019}  {Tobias Sebastian Finn}

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
import xarray as xr

import torch

# Internal modules
from pytassim.model.integration import RK4Integrator
from pytassim.model.lorenz_96 import Lorenz96, forward_model


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

rnd = np.random.RandomState(42)


class TestForwardModel(unittest.TestCase):
    def setUp(self):
        self.model = Lorenz96()
        self.integrator = RK4Integrator(self.model)
        self.all_steps = np.arange(10)
        self.start_point = 2
        self.start_state = torch.from_numpy(
            rnd.normal(scale=0.1, size=(1, 1, 40,))
        )

    def test_forward_models_returns_xarray(self):
        ret_arr = forward_model(self.all_steps, self.start_point,
                                self.start_state, self.integrator)
        self.assertIsInstance(ret_arr, xr.DataArray)

    def test_start_state_is_set_as_first_state(self):
        self.start_point = 0
        out_states = []
        curr_state = self.start_state
        for i in self.all_steps:
            curr_state = self.integrator.integrate(curr_state)
            out_states.append(curr_state.numpy())
        out_states = np.array(out_states)
        ret_arr = forward_model(self.all_steps, self.start_point,
                                self.start_state, self.integrator)
        for i in self.all_steps:
            np.testing.assert_equal(
                out_states[i].squeeze(),
                ret_arr.isel(time=i).values.squeeze()
            )

    def test_array_has_four_dimensions_depending_on_start_state(self):
        self.start_state = torch.from_numpy(
            rnd.normal(scale=0.1, size=(1, 50, 150,))
        )
        ret_arr = forward_model(self.all_steps, self.start_point,
                                self.start_state, self.integrator)
        self.assertEqual(len(ret_arr.ensemble), 50)
        self.assertEqual(len(ret_arr.grid), 150)


if __name__ == '__main__':
    unittest.main()
