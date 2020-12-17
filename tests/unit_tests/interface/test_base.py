#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12/5/18

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
from unittest.mock import patch, PropertyMock
import warnings

# External modules
import xarray as xr
import numpy as np
import pandas as pd
import torch

# Internal modules
from pytassim.interface.base import BaseAssimilation
from pytassim.state import StateError
from pytassim.observation import ObservationError
from pytassim.testing import dummy_obs_operator


logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class NewAssimilation(BaseAssimilation):
    def update_state(self, state, observations, pseudo_state, analysis_time):
        return state


class TestBaseAssimilation(unittest.TestCase):
    def setUp(self):
        self.algorithm = NewAssimilation()
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path)
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path)

    def test_dtype_returns_private_dtype(self):
        self.algorithm._dtype = torch.int
        self.assertEqual(self.algorithm.dtype, torch.int)

    def test_dtype_sets_private_dtype(self):
        self.algorithm._dtype = None
        self.algorithm.dtype = torch.int
        self.assertEqual(self.algorithm._dtype, torch.int)

    def test_dtype_raises_type_error_if_not_pytorch_dtype(self):
        with self.assertRaises(TypeError):
            self.algorithm.dtype = 123

    def test_device_returns_torch_device(self):
        self.assertIsInstance(self.algorithm.device, torch.device)

    def test_device_returns_gpu_if_gpu_true(self):
        self.algorithm.gpu = True
        self.assertEqual(self.algorithm.device, torch.device('cuda'))

    def test_device_returns_cpu_if_gpu_false(self):
        self.algorithm.gpu = False
        self.assertEqual(self.algorithm.device, torch.device('cpu'))

    def test_validate_state_calls_valid_from_state(self):
        with patch('pytassim.state.ModelState.valid',
                   new_callable=PropertyMock) as mock_state_valid:
            self.algorithm._validate_state(self.state)
        mock_state_valid.assert_called_once()

    def test_validate_state_raises_state_error_if_not_valid(self):
        self.state = self.state.rename(var_name='var_test')
        with self.assertRaises(StateError) as e:
            self.algorithm._validate_state(self.state)

    def test_validate_state_raises_type_error_if_not_dataarray(self):
        with self.assertRaises(TypeError):
            self.algorithm._validate_state(self.state.values)

    def test_validate_single_obs_calls_valid_from_obs(self):
        with patch('pytassim.observation.Observation.valid',
                   new_callable=PropertyMock) as mock_obs_valid:
            self.algorithm._validate_observations((self.obs, ))
            mock_obs_valid.assert_called_once()

    def test_validate_single_obs_raises_obs_error(self):
        self.obs = self.obs.rename(obs_grid_1='obs_grid')
        with self.assertRaises(ObservationError) as _:
            self.algorithm._validate_observations((self.obs, ))

    def test_validate_single_obs_tests_type(self):
        with self.assertRaises(TypeError) as _:
            self.algorithm._validate_observations((self.obs['observations'], ))

    def test_get_analysis_time_uses_analysis_time_if_valid(self):
        valid_time = pd.to_datetime(self.state.time[-1].to_pandas())
        returned_time = self.algorithm._get_analysis_time(
            self.state, analysis_time=valid_time
        )
        np.testing.assert_equal(valid_time, returned_time)

    def test_get_analysis_time_return_latest_time_if_none(self):
        valid_time = pd.to_datetime(self.state.time[-1].to_pandas())
        returned_time = self.algorithm._get_analysis_time(
            self.state, analysis_time=None
        )
        np.testing.assert_equal(valid_time, returned_time)

    def test_get_analysis_returns_nearest_time_if_not_valid(self):
        valid_time = pd.to_datetime(self.state.time[0].to_pandas())
        with self.assertWarns(UserWarning) as w:
            returned_time = self.algorithm._get_analysis_time(
                self.state, analysis_time='1991'
            )
        np.testing.assert_equal(valid_time, returned_time)

    def test_apply_obs_operator_filters_obs_wo_operator(self):
        obs_list = [self.obs, self.obs.copy()]
        obs_list[-1].obs.operator = dummy_obs_operator
        _, filtered_obs = self.algorithm._apply_obs_operator(self.state,
                                                             obs_list)
        self.assertIsInstance(filtered_obs, list)
        self.assertEqual(len(filtered_obs), 1)
        self.assertEqual(id(filtered_obs[0]), id(obs_list[-1]))

    def test_apply_applies_obs_operator_to_state(self):
        self.obs.obs.operator = dummy_obs_operator
        obs_list = [self.obs, ]
        obs_equivalent, _ = self.algorithm._apply_obs_operator(self.state,
                                                               obs_list)
        self.assertIsInstance(obs_equivalent, list)
        self.assertEqual(len(obs_equivalent), 1)
        xr.testing.assert_equal(self.obs.obs.operator(self.obs, self.state),
                                obs_equivalent[0])

    def test_obs_stacks_observations(self):
        stacked_obs = self.obs['observations'].stack(
            obs_id=['time', 'obs_grid_1']
        )
        stacked_obs = xr.concat((stacked_obs, stacked_obs), dim='obs_id')
        obs_list = [
            self.obs['observations'],
            self.obs['observations']
        ]
        returned_stacked_obs = self.algorithm._stack_obs(obs_list)
        xr.testing.assert_identical(stacked_obs, returned_stacked_obs)

    def test_obs_drops_multiindex_grid_if_multiindex(self):
        self.obs['obs_grid_1'] = pd.MultiIndex.from_product(
            [self.obs.indexes['obs_grid_1'], [0]],
            names=['test', 'test_1']
        )
        stacked_obs = self.obs['observations'].copy(deep=True)
        stacked_obs['obs_grid_1'] = pd.Index(
            stacked_obs.indexes['obs_grid_1'].values,
            tupleize_cols=False
        )
        stacked_obs = stacked_obs.stack(obs_id=['time', 'obs_grid_1'])
        returned_obs = self.algorithm._stack_obs([self.obs['observations']])
        xr.testing.assert_identical(returned_obs, stacked_obs)

    def test_assimilate_wo_obs_returns_state(self):
        with self.assertWarns(UserWarning):
            analysis = self.algorithm.assimilate(self.state, ())
        xr.testing.assert_identical(analysis, self.state)


if __name__ == '__main__':
    unittest.main()
