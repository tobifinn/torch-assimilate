#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12/7/18

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
from unittest.mock import patch

# External modules
import xarray as xr
import torch
import numpy as np

# Internal modules
import pytassim.state
import pytassim.observation
from pytassim.assimilation.filter.letkf import ETKFCorr
from pytassim.assimilation.filter.letkf import LETKFilter
from pytassim.assimilation.filter import etkf_core
from pytassim.testing import dummy_obs_operator, DummyLocalization


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestLETKF(unittest.TestCase):
    def setUp(self):
        self.algorithm = LETKFilter()
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        self.back_prec = self.algorithm._get_back_prec(len(self.state.ensemble))
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator

    def tearDown(self):
        self.state.close()
        self.obs.close()

    def test_wo_localization_letkf_equals_etkf(self):
        etkf = ETKFCorr()
        obs_tuple = (self.obs, self.obs)
        etkf_analysis = etkf.assimilate(self.state, obs_tuple)
        letkf_analysis = self.algorithm.assimilate(self.state, obs_tuple)
        xr.testing.assert_allclose(letkf_analysis, etkf_analysis)

    def test_update_state_calls_prepare(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        with patch('pytassim.assimilation.filter.letkf.LETKFilter._prepare',
                   return_value=prepared_states) as prepare_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple,
                                            self.state.time[-1].values)
        prepare_patch.assert_called_once_with(self.state, obs_tuple)

    def test_update_state_returns_valid_state(self):
        obs_tuple = (self.obs, self.obs)
        analysis = self.algorithm.update_state(self.state, obs_tuple,
                                               self.state.time[-1].values)
        self.assertTrue(analysis.state.valid)

    def test_dummy_localization_returns_equal_grids(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        obs_weights = (np.abs(prepared_states[-1]-10) < 10).astype(float)
        use_obs = obs_weights > 0

        localization = DummyLocalization()
        ret_use_obs, ret_weights = localization.localize_obs(
            10, prepared_states[-1]
        )

        np.testing.assert_equal(ret_use_obs, use_obs)
        np.testing.assert_equal(ret_weights, obs_weights)

    def test_update_state_uses_localization(self):
        self.algorithm.localization = DummyLocalization()
        ana_time = self.state.time[-1].values
        nr_grid_points = len(self.state.grid)
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        obs_weights = (np.abs(prepared_states[-1]-10) < 10).astype(float)
        use_obs = obs_weights > 0
        with patch('pytassim.testing.dummy.DummyLocalization.localize_obs',
                  return_value=(use_obs, obs_weights)) as loc_patch:
           _ = self.algorithm.update_state(self.state, obs_tuple, ana_time)
        self.assertEqual(loc_patch.call_count, nr_grid_points)

    def test_wo_localization_letkf_equals_etkf_smoothing(self):
        etkf = ETKFCorr(smoother=True)
        self.algorithm.smoother = True
        obs_tuple = (self.obs, self.obs)
        etkf_analysis = etkf.assimilate(self.state, obs_tuple)
        letkf_analysis = self.algorithm.assimilate(self.state, obs_tuple)
        xr.testing.assert_allclose(letkf_analysis, etkf_analysis)

    def test_algorithm_works(self):
        self.algorithm.inf_factor = 1.1
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      ana_time)
        self.assertFalse(np.any(np.isnan(assimilated_state.values)))


if __name__ == '__main__':
    unittest.main()
