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
from pytassim.assimilation.filter.letkf import ETKFilter
from pytassim.assimilation.filter.letkf import LETKFilter
from pytassim.testing import dummy_obs_operator, DummyLocalization


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestLETKF(unittest.TestCase):
    def setUp(self):
        self.algorithm = LETKFilter()
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator

    def tearDown(self):
        self.state.close()
        self.obs.close()

    def test_wo_localization_letkf_equals_etkf(self):
        etkf = ETKFilter()
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

    def test_update_state_calls_gen_weights_grid_times(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        nr_grid_points = len(self.state.grid)
        prepared_states = [torch.tensor(s) for s in prepared_states]
        weights = self.algorithm._gen_weights(*prepared_states[:-1])
        with patch('pytassim.assimilation.filter.letkf.LETKFilter._gen_weights',
                   return_value=weights) as weight_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple,
                                            self.state.time[-1].values)
        self.assertEqual(weight_patch.call_count, nr_grid_points)

    def test_update_state_calls_apply_weights_grid_times(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        nr_grid_points = len(self.state.grid)
        prepared_states = [torch.tensor(s) for s in prepared_states[:-1]]
        weights = self.algorithm._gen_weights(*prepared_states)
        back_state = self.state.sel(time=[ana_time, ])
        localized_state = back_state.isel(grid=0)
        trg = 'pytassim.assimilation.filter.letkf.LETKFilter._apply_weights'
        with patch(trg, return_value=localized_state) as apply_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple, ana_time)
        self.assertEqual(apply_patch.call_count, nr_grid_points)
        for i, grid_ind in enumerate(self.state.grid):
            state_l = back_state.sel(grid=grid_ind)
            mean_l, perts_l = state_l.state.split_mean_perts()
            torch.testing.assert_allclose(apply_patch.call_args_list[i][0][0],
                                          weights[0])
            torch.testing.assert_allclose(apply_patch.call_args_list[i][0][1],
                                          weights[1])
            xr.testing.assert_equal(apply_patch.call_args_list[i][0][2],
                                    mean_l)
            xr.testing.assert_equal(apply_patch.call_args_list[i][0][3],
                                    perts_l)

    def test_update_state_returns_valid_state(self):
        obs_tuple = (self.obs, self.obs)
        analysis = self.algorithm.update_state(self.state, obs_tuple,
                                               self.state.time[-1].values)
        self.assertTrue(analysis.state.valid)

    def test_dummy_localization_returns_equal_grids(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        obs_weights = (prepared_states[-1] == 10).astype(float)
        in_loc = obs_weights > 0

        localization = DummyLocalization()
        localized_states = localization.localize_obs(10, *prepared_states)

        np.testing.assert_equal(localized_states[0], prepared_states[0][in_loc])
        np.testing.assert_equal(localized_states[1], prepared_states[1][in_loc])
        np.testing.assert_equal(localized_states[2],
                                prepared_states[2][in_loc, in_loc])
        np.testing.assert_equal(localized_states[3], obs_weights[in_loc])

    def test_update_state_uses_localization(self):
        self.algorithm.localization = DummyLocalization()
        ana_time = self.state.time[-1].values
        nr_grid_points = len(self.state.grid)
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        with patch('pytassim.testing.dummy.DummyLocalization.localize_obs',
                   return_value=prepared_states[:-1]) as loc_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple, ana_time)
        self.assertEqual(loc_patch.call_count, nr_grid_points)

    def test_wo_localization_letkf_equals_etkf_smoothing(self):
        etkf = ETKFilter(smoothing=True)
        self.algorithm.smoothing = True
        obs_tuple = (self.obs, self.obs)
        etkf_analysis = etkf.assimilate(self.state, obs_tuple)
        letkf_analysis = self.algorithm.assimilate(self.state, obs_tuple)
        xr.testing.assert_allclose(letkf_analysis, etkf_analysis)


if __name__ == '__main__':
    unittest.main()
