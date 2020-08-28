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


logging.basicConfig(level=logging.INFO)

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

    def test_valid_dim_checks_auxiliary_dims(self):
        self.assertTrue(self.obs_ds.obs._valid_dims)
        self.obs_ds = self.obs_ds.rename({'obs_grid_2': 'test_2'})
        self.assertFalse(self.obs_ds.obs._valid_dims)

    def test_valid_obs_checks_the_last_two_dims_of_obs_array(self):
        self.assertTrue(self.obs_ds.obs._valid_obs)
        obs_ds = self.obs_ds.rename({'obs_grid_1': 'test_1'})
        self.assertFalse(obs_ds.obs._valid_obs)
        self.obs_ds['observations'] = self.obs_ds['observations'].T
        self.assertFalse(self.obs_ds.obs._valid_obs)

    def test_valid_cov_checks_last_two_dims_of_cov_array(self):
        self.assertTrue(self.obs_ds.obs._valid_cov_corr)
        obs_ds = self.obs_ds.rename({'obs_grid_1': 'test_1'})
        self.assertFalse(obs_ds.obs._valid_cov_corr)
        self.obs_ds['covariance'] = self.obs_ds['covariance'].T
        self.assertFalse(self.obs_ds.obs._valid_cov_corr)

    def test_valid_cov_checks_last_shape_of_dims(self):
        self.assertTrue(self.obs_ds.obs._valid_cov_corr)
        obs_ds = self.obs_ds.copy()
        obs_ds = obs_ds.isel(obs_grid_2=slice(0, 2))
        self.assertFalse(obs_ds.obs._valid_cov_corr)
        obs_ds = self.obs_ds.copy()
        obs_ds = obs_ds.isel(obs_grid_1=slice(0, 2))
        self.assertFalse(obs_ds.obs._valid_cov_corr)

    def test_valid_cov_checks_grid_dim_values(self):
        self.assertTrue(self.obs_ds.obs._valid_cov_corr)
        obs_ds = self.obs_ds.copy()
        obs_ds['obs_grid_2'] = obs_ds['obs_grid_2'] + 10
        self.assertFalse(obs_ds.obs._valid_cov_corr)

    def test_valid_cov_uncorr_checks_dim_order(self):
        self.assertFalse(self.obs_ds.obs._valid_cov_uncorr)
        self.obs_ds['covariance'] = xr.DataArray(
            np.diag(self.obs_ds['covariance'].values),
            coords={
                'obs_grid_1': self.obs_ds.obs_grid_1
            },
            dims=['obs_grid_1']
        )
        self.assertTrue(self.obs_ds.obs._valid_cov_uncorr)

    def test_correlated_property_checks_if_obs_grid_2_available(self):
        self.assertTrue(self.obs_ds.obs.correlated)
        self.obs_ds['covariance'] = xr.DataArray(
            np.diag(self.obs_ds['covariance'].values),
            coords={
                'obs_grid_1': self.obs_ds.obs_grid_1
            },
            dims=['obs_grid_1']
        )
        self.assertFalse(self.obs_ds.obs.correlated)

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

    def test_valid_array_uses_uncorrelated_for_uncorrelated_cov(self):
        self.obs_ds['covariance'] = xr.DataArray(
            np.diag(self.obs_ds['covariance'].values),
            coords={
                'obs_grid_1': self.obs_ds.obs_grid_1
            },
            dims=['obs_grid_1']
        )
        self.assertTrue(self.obs_ds.obs._valid_arrays)

    def test_valid_checks_if_vars_are_available(self):
        self.assertTrue(self.obs_ds.obs.valid)
        del self.obs_ds['covariance']
        self.assertFalse(self.obs_ds.obs.valid)

    def test_operator_raises_notimplemented(self):
        with self.assertRaises(NotImplementedError):
            _ = self.obs_ds.obs.operator(self.obs_ds, self.state)


if __name__ == '__main__':
    unittest.main()
