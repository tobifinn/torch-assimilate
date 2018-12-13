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

# Internal modules
from pytassim.assimilation.base import BaseAssimilation
from pytassim.state import StateError
from pytassim.observation import ObservationError
from pytassim.testing import dummy_update_state, dummy_obs_operator


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestBaseAssimilation(unittest.TestCase):
    def setUp(self):
        self.algorithm = BaseAssimilation()
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path)
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path)

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
            self.algorithm._validate_single_obs(self.obs)
            mock_obs_valid.assert_called_once()

    def test_validate_single_obs_raises_obs_error(self):
        self.obs = self.obs.rename(obs_grid_1='obs_grid')
        with self.assertRaises(ObservationError) as e:
            self.algorithm._validate_single_obs(self.obs)

    def test_validate_single_obs_tests_type(self):
        with self.assertRaises(TypeError) as e:
            self.algorithm._validate_single_obs(self.obs['observations'])

    def test_validate_multi_observations_calls_valid_single_obs(self):
        observations = (self.obs, self.obs)
        with patch('pytassim.assimilation.base.BaseAssimilation.'
                   '_validate_single_obs') as single_obs_patch:
            self.algorithm._validate_observations(observations)
            self.assertEqual(single_obs_patch.call_count, 2)

    def test_validate_single_observations_calls_valid_single_obs(self):
        with patch('pytassim.assimilation.base.BaseAssimilation.'
                   '_validate_single_obs') as single_obs_patch:
            self.algorithm._validate_observations(self.obs)
        single_obs_patch.assert_called_once_with(self.obs)

    def test_get_analysis_time_uses_analysis_time_if_valid(self):
        valid_time = self.state.time[-1]
        returned_time = self.algorithm._get_analysis_time(
            self.state, analysis_time=valid_time
        )
        np.testing.assert_equal(valid_time.values, returned_time)

    def test_get_analysis_time_return_latest_time_if_none(self):
        valid_time = self.state.time[-1]
        returned_time = self.algorithm._get_analysis_time(
            self.state, analysis_time=None
        )
        np.testing.assert_equal(valid_time.values, returned_time)

    def test_get_analysis_returns_nearest_time_if_not_valid(self):
        valid_time = self.state.time[0]
        with self.assertWarns(UserWarning) as w:
            returned_time = self.algorithm._get_analysis_time(
                self.state, analysis_time='1991'
            )
        np.testing.assert_equal(valid_time.values, returned_time)

    @patch('pytassim.assimilation.base.BaseAssimilation.update_state',
           side_effect=dummy_update_state, autospec=True)
    def test_assimilate_uses_latest_state_time(self, _):
        self.algorithm.smoother = True
        latest_time = self.state.time[-1]
        with patch('pytassim.assimilation.base.BaseAssimilation.'
                   '_get_analysis_time', return_value=latest_time) as time_mock:
            _ = self.algorithm.assimilate(self.state, self.obs, None)
        time_mock.assert_called_once_with(self.state, None)

    @patch('pytassim.assimilation.base.BaseAssimilation.update_state',
           side_effect=dummy_update_state, autospec=True)
    def test_assimilate_validates_state(self, _):
        with patch('pytassim.assimilation.base.BaseAssimilation.'
                   '_validate_state') as valid_mock:
            _ = self.algorithm.assimilate(self.state, self.obs, None)
            valid_mock.assert_called()
        xr.testing.assert_equal(valid_mock.call_args_list[0][0][0], self.state)

    @patch('pytassim.assimilation.base.BaseAssimilation.update_state',
           side_effect=dummy_update_state, autospec=True)
    def test_assimilate_validates_observations(self, _):
        with patch('pytassim.assimilation.base.BaseAssimilation.'
                   '_validate_observations') as valid_mock:
            _ = self.algorithm.assimilate(self.state, self.obs, None)
            valid_mock.assert_called()
        xr.testing.assert_equal(valid_mock.call_args[0][0][0], self.obs)

    @patch('pytassim.assimilation.base.BaseAssimilation.update_state',
           side_effect=dummy_update_state, autospec=True)
    def test_assimilate_calls_update_state(self, update_mock):
        self.algorithm.smoother = True
        _ = self.algorithm.assimilate(self.state, self.obs, None)
        latest_time = self.state.time[-1]
        update_mock.assert_called_once()
        xr.testing.assert_equal(update_mock.call_args[0][1], self.state)
        xr.testing.assert_equal(update_mock.call_args[0][2][0], self.obs,)
        np.testing.assert_equal(update_mock.call_args[0][3], latest_time.values)

    @patch('pytassim.assimilation.base.BaseAssimilation.update_state',
           side_effect=dummy_update_state, autospec=True)
    def test_assimilate_converts_single_obs_to_tuple(self, update_mock):
        self.algorithm.smoother = True
        _ = self.algorithm.assimilate(self.state, self.obs, None)
        self.assertIsInstance(update_mock.call_args[0][2], tuple)

    @patch('pytassim.assimilation.base.BaseAssimilation.update_state',
           side_effect=dummy_update_state, autospec=True)
    def test_assimilate_validates_analysis(self, _):
        analysis = dummy_update_state(self.algorithm, self.state, self.obs,
                                      self.state.time[-1])
        with patch('pytassim.assimilation.base.BaseAssimilation.'
                   '_validate_state') as valid_mock:
            _ = self.algorithm.assimilate(self.state, self.obs, None)
        self.assertEqual(valid_mock.call_count, 2)
        xr.testing.assert_equal(valid_mock.call_args_list[1][0][0], analysis)

    @patch('pytassim.assimilation.base.BaseAssimilation.update_state',
           side_effect=dummy_update_state, autospec=True)
    def test_assimilate_returns_analysis(self, _):
        analysis = dummy_update_state(self.algorithm, self.state, self.obs,
                                      self.state.time[-1])
        returned_analysis = self.algorithm.assimilate(self.state, self.obs,
                                                      None)
        xr.testing.assert_equal(analysis, returned_analysis)

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
        xr.testing.assert_equal(self.obs.obs.operator(self.state),
                                obs_equivalent[0])

    @patch('pytassim.assimilation.base.BaseAssimilation.update_state',
           side_effect=dummy_update_state, autospec=True)
    def test_assimilate_wo_obs_returns_state(self, _):
        with self.assertWarns(UserWarning):
            analysis = self.algorithm.assimilate(self.state, ())
        xr.testing.assert_identical(analysis, self.state)


if __name__ == '__main__':
    unittest.main()
