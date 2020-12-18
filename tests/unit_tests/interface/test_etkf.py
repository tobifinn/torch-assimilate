#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12/6/18

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
from unittest.mock import patch, MagicMock
import re

# External modules
import xarray as xr
import numpy as np
import pandas as pd

import torch
import torch.jit
import torch.nn
import torch.sparse

import scipy.linalg
import scipy.linalg.blas

# Internal modules
import pytassim.state
import pytassim.observation
from pytassim.interface.etkf import ETKF
from pytassim.testing import dummy_obs_operator, if_gpu_decorator


logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestETKF(unittest.TestCase):
    def setUp(self):
        self.algorithm = ETKF()
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator

    def tearDown(self):
        self.state.close()
        self.obs.close()

    def test_inf_factor_sets_core_module(self):
        old_id = id(self.algorithm._core_module)
        self.algorithm.inf_factor = 3.2
        self.assertNotEqual(id(self.algorithm._core_module), old_id)
        self.assertEqual(self.algorithm._core_module.inf_factor, 3.2)

    def test_algorithm_works(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      None, ana_time)
        self.assertFalse(np.any(np.isnan(assimilated_state.values)))

    def test_algorithm_works_dask(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        analysis = self.algorithm.assimilate(self.state, obs_tuple, None,
                                             ana_time)
        chunked_state = self.state.chunk({'grid': 1})
        chunked_analysis = self.algorithm.assimilate(
            chunked_state, obs_tuple, None, ana_time
        )
        chunked_analysis = chunked_analysis.load()
        np.testing.assert_allclose(
            chunked_analysis.values, analysis.values, rtol=1E-10, atol=1E-10
        )

    def test_algorithm_works_time(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        no_time = self.algorithm.assimilate(self.state, obs_tuple, None,
                                            ana_time)
        self.obs['covariance'] = self.obs['covariance'].expand_dims(
            time=self.obs['time']
        )
        obs_tuple = (self.obs, self.obs.copy())
        with_time = self.algorithm.assimilate(self.state, obs_tuple,
                                              None, ana_time)
        xr.testing.assert_identical(with_time, no_time)

    def test_etkf_returns_right_analysis(self):
        sliced_state = self.state.isel(time=[0])
        sliced_obs = self.obs.isel(time=[0])
        sliced_obs.obs.operator = self.obs.obs.operator
        ens_obs = sliced_obs.obs.operator(sliced_obs, sliced_state)
        ens_mean, ens_perts = ens_obs.state.split_mean_perts()
        innovation = sliced_obs['observations']-ens_mean

        norm_innov = sliced_obs.obs.mul_rcinv(innovation)
        norm_innov = torch.from_numpy(norm_innov.values).to(
            self.algorithm.dtype
        ).view(1, -1)

        norm_perts = sliced_obs.obs.mul_rcinv(ens_perts)
        norm_perts = norm_perts.transpose('ensemble', 'time', 'obs_grid_1')
        norm_perts = torch.from_numpy(norm_perts.values).to(
            self.algorithm.dtype
        ).view(10, -1)

        weights = self.algorithm.module(norm_perts, norm_innov)[0].numpy()
        weights = xr.DataArray(
            weights,
            coords={
                'ensemble': self.state.indexes['ensemble'],
                'ensemble_new': self.state.indexes['ensemble']
            },
            dims=['ensemble', 'ensemble_new']
        )
        analysis = self.algorithm._apply_weights(sliced_state, weights)
        ret_analysis = self.algorithm.assimilate(state=sliced_state,
                                                 observations=sliced_obs)
        xr.testing.assert_identical(analysis, ret_analysis)

    def test_filter_uses_state_as_pseudo_state_if_no_pseudo(self):
        right_analysis = self.algorithm.assimilate(self.state, self.obs,
                                                   self.state)
        ret_analysis = self.algorithm.assimilate(self.state, self.obs)
        xr.testing.assert_equal(right_analysis, ret_analysis)


if __name__ == '__main__':
    unittest.main()
