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
from unittest.mock import MagicMock

# External modules
import xarray as xr
import numpy as np

# Internal module
from pytassim.observation import Observation
from pytassim.obs_ops.base_ops import BaseOperator


logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestBaseOperator(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path)
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path)
        self.operator = BaseOperator()

    def test_call_calls_obs_op(self):
        self.operator.obs_op = MagicMock()
        self.operator(self.obs, self.state, 1, test=2)
        self.operator.obs_op.assert_called_once_with(self.state, 1, test=2)

    def test_get_obs_method_uses_obs_op(self):
        self.operator.obs_op = MagicMock(return_value=self.state)
        _ = self.operator(self.obs, self.state)
        self.operator.obs_op.assert_called_once_with(self.state)

    def test_get_obs_method_renames_grid(self):
        self.operator.obs_op = MagicMock(return_value=self.state)
        pseudo_obs = self.operator(self.obs, self.state)
        np.testing.assert_equal(self.state.values, pseudo_obs.values)
        self.assertEqual(pseudo_obs.dims[-1], 'obs_grid_1')

    def test_get_obs_method_sets_new_time(self):
        self.operator.obs_op = MagicMock(return_value=self.state)
        self.obs['time'] = self.obs['time'] + 1
        pseudo_obs = self.operator(self.obs, self.state)
        np.testing.assert_equal(pseudo_obs.time.values, self.obs.time.values)

    def test_get_obs_method_sets_new_grid(self):
        self.operator.obs_op = MagicMock(return_value=self.state)
        self.obs['obs_grid_1'] = self.obs['obs_grid_1'] + 1
        pseudo_obs = self.operator(self.obs, self.state)
        np.testing.assert_equal(pseudo_obs.obs_grid_1.values,
                                self.obs.obs_grid_1.values)


if __name__ == '__main__':
    unittest.main()
