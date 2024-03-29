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

# External modules
import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da

import torch

# Internal modules
from pytassim.interface.etkf import ETKF
from pytassim.interface.letkf import LETKF
from pytassim.localization import GaspariCohn
from pytassim.testing import dummy_obs_operator, if_gpu_decorator


logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestLETKF(unittest.TestCase):
    def setUp(self):
        self.algorithm = LETKF()
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator

    def tearDown(self):
        self.state.close()
        self.obs.close()

    def test_wo_localization_letkf_equals_etkf(self):
        etkf = ETKF()
        obs_tuple = (self.obs, self.obs)
        etkf_analysis = etkf.assimilate(self.state, obs_tuple)
        letkf_analysis = self.algorithm.assimilate(self.state, obs_tuple)
        xr.testing.assert_allclose(letkf_analysis, etkf_analysis,
                                   rtol=1E-10, atol=1E-10)

    def test_update_state_returns_valid_state(self):
        obs_tuple = (self.obs, self.obs)
        analysis = self.algorithm.update_state(
            self.state, obs_tuple, self.state, self.state.time[-1].values
        )
        self.assertTrue(analysis.state.valid)

    def test_algorithm_localized_works_multiindex_grid(self):
        self.state['grid'] = pd.MultiIndex.from_product(
            (np.arange(40), [0,]), names=['grid_point', 'height']
        )
        self.algorithm.localization = GaspariCohn(
            (1., 1.), dist_func=lambda x, y: np.zeros(y.shape[0])
        )
        self.algorithm.chunksize = 10
        obs_tuple = (self.obs, self.obs)
        etkf = ETKF()
        etkf_analysis = etkf.assimilate(self.state, obs_tuple)
        letkf_analysis = self.algorithm.assimilate(self.state, obs_tuple)
        xr.testing.assert_allclose(letkf_analysis, etkf_analysis,
                                   rtol=1E-10, atol=1E-10)

    def test_algorithm_localized_works(self):
        self.algorithm.localization = GaspariCohn(
            (1., 1.), dist_func=lambda x, y: np.zeros(y.shape[0])
        )
        self.algorithm.chunksize = 10
        obs_tuple = (self.obs, self.obs)
        etkf = ETKF()
        etkf_analysis = etkf.assimilate(self.state, obs_tuple)
        letkf_analysis = self.algorithm.assimilate(self.state, obs_tuple)
        xr.testing.assert_allclose(letkf_analysis, etkf_analysis,
                                   rtol=1E-10, atol=1E-10)

    def test_algorithm_localized_right(self):
        def dist_func(x, y):
            diff = x - y
            abs_diff = diff['obs_grid_1'].abs().values
            return abs_diff,
        self.algorithm.localization = GaspariCohn(
            (10.,), dist_func=dist_func
        )
        sliced_state = self.state.isel(time=[0])
        sliced_obs = self.obs.isel(time=[0])
        sliced_obs.obs.operator = self.obs.obs.operator
        ens_obs = sliced_obs.obs.operator(sliced_obs, sliced_state)
        norm_innov, norm_perts = self.algorithm._get_obs_space_variables(
            [ens_obs], [sliced_obs]
        )
        obs_info = self.algorithm._extract_obs_information(norm_innov)
        grid_index, state_info = self.algorithm._extract_state_information(
            sliced_state
        )

        torch_innov = torch.from_numpy(norm_innov.values).to(
            self.algorithm.dtype
        ).view(1, -1)
        torch_perts = torch.from_numpy(norm_perts.values).to(
            self.algorithm.dtype
        ).view(10, -1)

        weights = []
        for curr_info in state_info.values:
            luse, lweights = self.algorithm.localization.localize_obs(
                curr_info, obs_info
            )
            lweights = np.sqrt(lweights[luse])
            curr_innov = torch_innov[..., luse] * lweights
            curr_perts = torch_perts[..., luse] * lweights
            curr_weights = self.algorithm.core_module(curr_perts, curr_innov)
            weights.append(curr_weights.numpy())
        weights = np.stack(weights, axis=0)

        weights = xr.DataArray(
            weights,
            coords={
                'grid': grid_index,
                'ensemble': sliced_state.indexes['ensemble'],
                'ensemble_new': sliced_state.indexes['ensemble']
            },
            dims=['grid', 'ensemble', 'ensemble_new']
        )
        right_analysis = self.algorithm._apply_weights(sliced_state, weights)
        ret_analysis = self.algorithm.assimilate(sliced_state, sliced_obs)
        xr.testing.assert_allclose(right_analysis, ret_analysis,
                                   rtol=1E-10, atol=1E-10)

    @if_gpu_decorator
    def test_algorithm_works_gpu(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        self.algorithm.gpu = True
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      None, ana_time)
        self.assertFalse(np.any(np.isnan(assimilated_state.values)))

    def test_chunks_return_chunksize(self):
        self.assertDictEqual(
            {'grid': self.algorithm.chunksize}, self.algorithm.chunks
        )

    def test_load_weights_returns_dask_array_if_grid_and_chunksize(self):
        weights = np.random.binomial(n=1, p=0.5, size=(10, 10, 40))
        weights = weights / (weights.sum(axis=0, keepdims=True) + 1E-9)
        weights = xr.DataArray(
            weights,
            coords={
                'ensemble': self.state['ensemble'].values,
                'ensemble_new': self.state['ensemble'].values,
                'grid': self.state.indexes['grid']
            },
            dims=['ensemble', 'ensemble_new', 'grid']
        )
        self.algorithm.chunksize = 20
        self.algorithm.weight_save_path = '/tmp/test.nc'
        self.algorithm.store_weights(weights)
        try:
            loaded_weights = self.algorithm.load_weights()
            self.assertIsInstance(loaded_weights.data, da.Array)
            self.assertTupleEqual(
                loaded_weights.data.chunksize, (10, 10, 20)
            )
            xr.testing.assert_identical(loaded_weights, weights)
        finally:
            if os.path.isfile(self.algorithm.weight_save_path):
                os.remove(self.algorithm.weight_save_path)


if __name__ == '__main__':
    unittest.main()
