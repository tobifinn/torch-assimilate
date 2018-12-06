#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 3/19/18

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
import types

# External modules
import numpy as np
import xarray as xr

# Internal modules
from pytassim.observation import Observation
from pytassim.testing import dummy_obs_operator


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


rnd = np.random.RandomState(42)


class TestObsSubset(unittest.TestCase):
    def setUp(self):
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs_ds = xr.open_dataset(obs_path)
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path)

    def test_xr_dataset_has_accessor(self):
        self.assertTrue(hasattr(self.obs_ds, 'obs'))

    def test_xr_ds_is_the_same_as_given_ds(self):
        xr.testing.assert_identical(self.obs_ds, self.obs_ds.obs.ds)

    def test_valid_is_bool_property(self):
        self.assertIsInstance(self.obs_ds.obs.valid, bool)

    def test_valid_checks_if_time_is_given(self):
        self.assertTrue(self.obs_ds.obs.valid)
        self.assertFalse(self.obs_ds.isel(time=0).obs.valid)

    def test_valid_checks_if_grid_points_1_exists(self):
        self.assertTrue(self.obs_ds.obs.valid)
        self.obs_ds = self.obs_ds.rename({'obs_grid_1': 'test_1'})
        self.assertFalse(self.obs_ds.obs.valid)

    def test_valid_checks_if_grid_points_2_exists(self):
        self.assertTrue(self.obs_ds.obs.valid)
        self.obs_ds = self.obs_ds.rename({'obs_grid_2': 'test_1'})
        self.assertFalse(self.obs_ds.obs.valid)

    def test_valid_obs_checks_the_last_two_dims_of_obs_array(self):
        self.assertTrue(self.obs_ds.obs._valid_obs)
        obs_ds = self.obs_ds.rename({'obs_grid_1': 'test_1'})
        self.assertFalse(obs_ds.obs._valid_obs)
        self.obs_ds['observations'] = self.obs_ds['observations'].T
        self.assertFalse(self.obs_ds.obs._valid_obs)

    def test_valid_cov_checks_last_two_dims_of_cov_array(self):
        self.assertTrue(self.obs_ds.obs._valid_cov)
        obs_ds = self.obs_ds.rename({'obs_grid_1': 'test_1'})
        self.assertFalse(obs_ds.obs._valid_cov)
        self.obs_ds['covariance'] = self.obs_ds['covariance'].T
        self.assertFalse(self.obs_ds.obs._valid_cov)

    def test_valid_cov_checks_last_shape_of_dims(self):
        self.assertTrue(self.obs_ds.obs._valid_cov)
        obs_ds = self.obs_ds.copy()
        obs_ds = obs_ds.isel(obs_grid_2=slice(0, 2))
        self.assertFalse(obs_ds.obs._valid_cov)
        obs_ds = self.obs_ds.copy()
        obs_ds = obs_ds.isel(obs_grid_1=slice(0, 2))
        self.assertFalse(obs_ds.obs._valid_cov)

    def test_valid_cov_checks_grid_dim_values(self):
        self.assertTrue(self.obs_ds.obs._valid_cov)
        obs_ds = self.obs_ds.copy()
        obs_ds['obs_grid_2'] = obs_ds['obs_grid_2'] + 10
        self.assertFalse(obs_ds.obs._valid_cov)

    def test_valid_arrays_checks_if_arrays_available(self):
        self.assertTrue(self.obs_ds.obs._valid_arrays)
        obs_ds = self.obs_ds.copy()
        del obs_ds['covariance']
        self.assertFalse(obs_ds.obs._valid_arrays)
        obs_ds = self.obs_ds.copy()
        del obs_ds['observations']
        self.assertFalse(obs_ds.obs._valid_arrays)

    def test_valid_arrays_checks_dims(self):
        self.assertTrue(self.obs_ds.obs._valid_arrays)
        obs_ds = self.obs_ds.copy()
        obs_ds = obs_ds.isel(obs_grid_1=slice(0, 2))
        self.assertFalse(obs_ds.obs._valid_arrays)
        obs_ds = self.obs_ds.copy()
        obs_ds = obs_ds.isel(obs_grid_2=slice(0, 2))
        self.assertFalse(obs_ds.obs._valid_arrays)

    def test_valid_checks_if_vars_are_available(self):
        self.assertTrue(self.obs_ds.obs.valid)
        del self.obs_ds['covariance']
        self.assertFalse(self.obs_ds.obs.valid)

    def test_operator_returns_private_operator(self):
        self.assertTrue(callable(self.obs_ds.obs.operator))
        with self.assertRaises(NotImplementedError):
            self.obs_ds.obs.operator(1)

    def test_operator_setter_sets_instance_method(self):
        self.assertIsInstance(self.obs_ds.obs._operator, types.MethodType)
        self.obs_ds.obs.operator = dummy_obs_operator
        self.assertIsInstance(self.obs_ds.obs._operator, types.MethodType)

    def test_operator_setter_sets_private_operator(self):
        self.obs_ds.obs.operator = dummy_obs_operator
        pseudo_obs = dummy_obs_operator(self.obs_ds.obs, self.state)
        return_value = self.obs_ds.obs.operator(self.state)
        xr.testing.assert_equal(pseudo_obs, return_value)

    def test_operator_checks_if_callable(self):
        with self.assertRaises(TypeError):
            self.obs_ds.obs.operator = 1

    def test_operator_checks_if_state_is_single_argument(self):
        def obs_operator(cls, state, time):
            return state

        with self.assertRaises(ValueError):
            self.obs_ds.obs.operator = obs_operator

        def obs_operator(cls):
            return None

        with self.assertRaises(ValueError):
            self.obs_ds.obs.operator = obs_operator

    def test_operator_sets_only_instance(self):
        obs_list = [self.obs_ds, self.obs_ds.copy()]
        self.assertNotEqual(id(obs_list[0].obs), id(obs_list[-1].obs))
        obs_list[-1].obs.operator = dummy_obs_operator
        pseudo_obs = dummy_obs_operator(obs_list[-1].obs, self.state)
        with self.assertRaises(NotImplementedError):
            self.obs_ds.obs.operator(self.state)
        returned_value = obs_list[-1].obs.operator(self.state)
        xr.testing.assert_equal(returned_value, pseudo_obs)


if __name__ == '__main__':
    unittest.main()
