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
        self.algorithm.inf_factor = torch.tensor(3.2)
        self.assertNotEqual(id(self.algorithm._core_module), old_id)
        torch.testing.assert_allclose(self.algorithm.core_module.inf_factor,
                                      3.2)

    def test_inf_factor_can_set_parameter(self):
        new_inf_factor = torch.nn.Parameter(torch.tensor(3.2))
        self.algorithm.inf_factor = new_inf_factor
        self.assertEqual(self.algorithm.core_module.inf_factor, new_inf_factor)

    def test_float_inf_factor_gets_converted_into_tensor(self):
        self.algorithm.inf_factor = 3.2
        self.assertIsInstance(
            self.algorithm.core_module.inf_factor, torch.Tensor
        )
        torch.testing.assert_allclose(
            self.algorithm.core_module.inf_factor, 3.2
        )

    def test_algorithm_works(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      None, ana_time)
        self.assertFalse(np.any(np.isnan(assimilated_state.values)))

    @if_gpu_decorator
    def test_algorithm_works_gpu(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        self.algorithm.gpu = True
        self.algorithm.inf_factor = torch.nn.Parameter(torch.tensor(2.0))
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

        weights = self.algorithm.core_module(norm_perts, norm_innov).numpy()
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

    def test_get_pseudo_obs_returns_pseudo_obs_if_given(self):
        pseudo_obs = self.algorithm.get_pseudo_state(
            pseudo_state=self.state+1,
            state=self.state,
            weights=self.algorithm.generate_prior_weights(
                self.state['ensemble'].values
            )
        )
        xr.testing.assert_identical(pseudo_obs, self.state+1)

    def test_get_pseudo_obs_propagates_model_if_no_pseudo_and_given_model(self):
        self.algorithm.forward_model = MagicMock()
        self.algorithm.forward_model.return_value = (self.state, self.state+5)
        pseudo_obs = self.algorithm.get_pseudo_state(
            pseudo_state=None,
            state=self.state,
            weights=self.algorithm.generate_prior_weights(
                self.state['ensemble'].values
            )
        )
        xr.testing.assert_identical(pseudo_obs, self.state+5)
        self.algorithm.forward_model.assert_called_once()

    def test_get_pseudo_obs_sets_state_to_pseudo_if_no_pseudo_and_model(self):
        self.algorithm.forward_model = None
        pseudo_obs = self.algorithm.get_pseudo_state(
            pseudo_state=None,
            state=self.state+1,
            weights=self.algorithm.generate_prior_weights(
                self.state['ensemble'].values
            )
        )
        xr.testing.assert_identical(pseudo_obs, self.state+1)

    def test_update_state_calls_generate_prior_weights(self):
        weights = self.algorithm.generate_prior_weights(np.arange(10))
        with patch(
                'pytassim.interface.base.BaseAssimilation.'
                'generate_prior_weights',
                return_value=weights
        ) as prior_patch:
            _ = self.algorithm.update_state(
                state=self.state,
                observations=(self.obs, ),
                pseudo_state=self.state,
                analysis_time=self.state.indexes['time'][0]
            )
        prior_patch.assert_called_once()
        np.testing.assert_equal(
            prior_patch.call_args[0][0],
            self.state['ensemble'].values
        )

    def test_update_state_calls_get_pseudo_state(self):
        weights = self.algorithm.generate_prior_weights(np.arange(10))
        with patch(
                'pytassim.interface.base.BaseAssimilation.'
                'generate_prior_weights',
                return_value=weights
        ), patch(
                'pytassim.interface.base.BaseAssimilation.get_pseudo_state',
                return_value=self.state
        ) as pseudo_patch:
            _ = self.algorithm.update_state(
                state=self.state,
                observations=(self.obs, ),
                pseudo_state=self.state,
                analysis_time=self.state.indexes['time'][0]
            )
        pseudo_patch.assert_called_once_with(
            pseudo_state=self.state,
            state=self.state,
            weights=weights
        )

    def test_chunks_return_None(self):
        self.assertIsNone(self.algorithm.chunks)

    def test_weights_are_stored_if_store_path_is_specified(self):
        self.algorithm.weight_save_path = None
        weights = self.algorithm.generate_prior_weights(np.arange(10))
        with patch(
                'pytassim.interface.base.BaseAssimilation.'
                'load_weights',
                return_value=weights
        ), patch(
                'pytassim.interface.base.BaseAssimilation.store_weights',
                return_value=None
        ) as store_patch:
            _ = self.algorithm.update_state(
                state=self.state,
                observations=(self.obs, ),
                pseudo_state=self.state,
                analysis_time=self.state.indexes['time'][0]
            )
            self.algorithm.weight_save_path = 'test.nc'
            _ = self.algorithm.update_state(
                state=self.state,
                observations=(self.obs, ),
                pseudo_state=self.state,
                analysis_time=self.state.indexes['time'][0]
            )
        store_patch.assert_called_once()

    def test_weights_are_loaded_if_store_path_is_specified(self):
        self.algorithm.weight_save_path = None
        weights = self.algorithm.generate_prior_weights(np.arange(10))
        with patch(
                'pytassim.interface.base.BaseAssimilation.'
                'load_weights',
                return_value=weights
        ) as load_patch, patch(
                'pytassim.interface.base.BaseAssimilation.store_weights',
                return_value=None
        ):
            _ = self.algorithm.update_state(
                state=self.state,
                observations=(self.obs, ),
                pseudo_state=self.state,
                analysis_time=self.state.indexes['time'][0]
            )
            self.algorithm.weight_save_path = 'test.nc'
            _ = self.algorithm.update_state(
                state=self.state,
                observations=(self.obs, ),
                pseudo_state=self.state,
                analysis_time=self.state.indexes['time'][0]
            )
        load_patch.assert_called_once()


if __name__ == '__main__':
    unittest.main()
