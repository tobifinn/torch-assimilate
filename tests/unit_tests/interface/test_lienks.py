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

# External modules
import xarray as xr
import numpy as np
import pandas as pd
import cloudpickle

import torch
import torch.jit
import torch.nn
import torch.sparse

# Internal modules
from pytassim.interface.lienks import LocalizedIEnKSTransform, \
    LocalizedIEnKSBundle
from pytassim.interface.ienks import IEnKSTransform, IEnKSBundle
from pytassim.interface.letkf import LETKF
from pytassim.interface.mixin_local import DomainLocalizedMixin
from pytassim.localization.gaspari_cohn import GaspariCohn
from pytassim.testing import dummy_obs_operator, if_gpu_decorator, \
    generate_random_weights


rnd = np.random.RandomState(42)

logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestLIEnKSTransform(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator
        self.weights = generate_random_weights(len(self.state['ensemble']))
        self.algorithm = LocalizedIEnKSTransform(
            model=lambda state, iter_num: (self.state + 1, self.state + 2)
        )

    def tearDown(self):
        self.state.close()
        self.obs.close()

    def test_isinstance_of_domain_localized_ienks_transform(self):
        self.assertIsInstance(self.algorithm, IEnKSTransform)
        self.assertIsInstance(self.algorithm, DomainLocalizedMixin)

    def test_localized_module_pickeable(self):
        cloudpickle.dumps(self.algorithm.core_module)
        cloudpickle.dumps(self.algorithm.module)
        cloudpickle.dumps(self.algorithm.localized_module)

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

    def test_with_no_localization_equals_ienks(self):
        self.algorithm.localization = None
        ienks = IEnKSTransform(model=self.algorithm.model,
                               tau=self.algorithm.tau)
        ens_obs = self.obs.obs.operator(self.obs, self.state)
        ienks_weights = ienks.estimate_weights(
            self.state.isel(time=[0]), self.weights, [self.obs], [ens_obs]
        )
        lienks_weights = self.algorithm.estimate_weights(
            self.state.isel(time=[0]), self.weights, [self.obs], [ens_obs]
        )

        lienks_weights = lienks_weights.mean('grid')
        xr.testing.assert_allclose(lienks_weights, ienks_weights)

    def test_lienks_with_linear_equals_letkf(self):
        def dist_func(x, y):
            diff = x - y
            abs_diff = diff['obs_grid_1'].abs().values
            return abs_diff,
        self.algorithm.localization = GaspariCohn(
            (10.,), dist_func=dist_func
        )

        def linear_model(state, iter_num):
            state = xr.concat([state,] * 3, dim='time')
            state['time'] = self.state['time'].values
            pseudo_state = state + 1
            return state, pseudo_state
        prior_state = self.state.isel(time=[0])
        propagated_state, pseudo_state = linear_model(prior_state, 0)

        letkf = LETKF(
            localization=self.algorithm.localization,
            inf_factor=1.0, smoother=True
        )
        letkf_analysis = letkf.assimilate(
            propagated_state, self.obs, pseudo_state,
            analysis_time=prior_state.time.values
        )

        self.algorithm.max_iter = 1
        self.algorithm.tau = 1.0
        self.algorithm.model = linear_model
        self.algorithm.smoother = True
        ienks_analysis = self.algorithm.assimilate(
            prior_state, self.obs, analysis_time=prior_state.time.values
        )
        xr.testing.assert_allclose(ienks_analysis, letkf_analysis)

    def test_lienks_with_linear_equals_letkf_multiindex_multi_iter(self):
        self.algorithm.weight_save_path = '/tmp/test.nc'
        self.state['grid'] = pd.MultiIndex.from_product(
            (np.arange(40), [0,]), names=['grid_point', 'height']
        )

        def dist_func(x, y):
            diff = x[1] - y['obs_grid_1']
            abs_diff = diff.abs().values
            return abs_diff,
        self.algorithm.localization = GaspariCohn(
            (10.,), dist_func=dist_func
        )

        def model(state, iter_num):
            state = xr.concat([
                state,
                np.sin(state),
                np.cos(state)
            ], dim='time')
            state['time'] = self.state['time'].values
            pseudo_state = state.compute() + 1
            return state, pseudo_state
        prior_state = self.state.isel(time=[0])

        self.algorithm.max_iter = 10
        self.algorithm.tau = 1.0
        self.algorithm.model = model
        self.algorithm.smoother = True

        logging.basicConfig(level=logging.DEBUG)
        ienks_analysis = self.algorithm.assimilate(
            prior_state, self.obs, analysis_time=prior_state.time.values
        )
        if os.path.isfile(self.algorithm.weight_save_path):
            os.remove(self.algorithm.weight_save_path)


    def test_lienks_with_linear_equals_letkf_multiindex(self):
        self.state['grid'] = pd.MultiIndex.from_product(
            (np.arange(40), [0,]), names=['grid_point', 'height']
        )

        def dist_func(x, y):
            diff = x[1] - y['obs_grid_1']
            abs_diff = diff.abs().values
            return abs_diff,
        self.algorithm.localization = GaspariCohn(
            (10.,), dist_func=dist_func
        )

        def linear_model(state, iter_num):
            state = xr.concat([state,] * 3, dim='time')
            state['time'] = self.state['time'].values
            pseudo_state = state + 1
            return state, pseudo_state
        prior_state = self.state.isel(time=[0])
        propagated_state, pseudo_state = linear_model(prior_state, 0)

        letkf = LETKF(
            localization=self.algorithm.localization,
            inf_factor=1.0, smoother=True
        )
        letkf_analysis = letkf.assimilate(
            propagated_state, self.obs, pseudo_state,
            analysis_time=prior_state.time.values
        )

        self.algorithm.max_iter = 1
        self.algorithm.tau = 1.0
        self.algorithm.model = linear_model
        self.algorithm.smoother = True
        ienks_analysis = self.algorithm.assimilate(
            prior_state, self.obs, analysis_time=prior_state.time.values
        )
        xr.testing.assert_allclose(ienks_analysis, letkf_analysis)


class TestLIEnKSBundle(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator
        self.weights = generate_random_weights(len(self.state['ensemble']))
        self.algorithm = LocalizedIEnKSBundle(
            model=lambda state, iter_num: (self.state + 1, self.state + 2)
        )

    def tearDown(self):
        self.state.close()
        self.obs.close()

    def test_isinstance_of_domain_localized_ienks_bundle(self):
        self.assertIsInstance(self.algorithm, IEnKSBundle)
        self.assertIsInstance(self.algorithm, DomainLocalizedMixin)

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

    def test_with_no_localization_equals_ienks(self):
        self.algorithm.localization = None
        ienks = IEnKSBundle(
            model=self.algorithm.model, tau=self.algorithm.tau,
            epsilon=self.algorithm.epsilon
        )
        ens_obs = self.obs.obs.operator(self.obs, self.state)
        ienks_weights = ienks.estimate_weights(
            self.state.isel(time=[0]), self.weights, [self.obs], [ens_obs]
        )
        lienks_weights = self.algorithm.estimate_weights(
            self.state.isel(time=[0]), self.weights, [self.obs], [ens_obs]
        )
        lienks_weights = lienks_weights.mean('grid')
        xr.testing.assert_allclose(lienks_weights, ienks_weights)

    def test_lienks_with_linear_equals_letkf(self):
        def dist_func(x, y):
            diff = x - y
            abs_diff = diff['obs_grid_1'].abs().values
            return abs_diff,
        self.algorithm.localization = GaspariCohn(
            (10.,), dist_func=dist_func
        )

        def linear_model(state, iter_num):
            state = xr.concat([state,] * 3, dim='time')
            state['time'] = self.state['time'].values
            pseudo_state = state + 1
            return state, pseudo_state
        prior_state = self.state.isel(time=[0])
        propagated_state, pseudo_state = linear_model(prior_state, 0)

        letkf = LETKF(
            localization=self.algorithm.localization,
            inf_factor=1.0, smoother=True
        )
        letkf_analysis = letkf.assimilate(
            propagated_state, self.obs, pseudo_state,
            analysis_time=prior_state.time.values
        )

        self.algorithm.max_iter = 1
        self.algorithm.tau = 1.0
        self.algorithm.epsilon = 1.0
        self.algorithm.model = linear_model
        self.algorithm.smoother = True
        ienks_analysis = self.algorithm.assimilate(
            prior_state, self.obs, analysis_time=prior_state.time.values
        )
        xr.testing.assert_allclose(ienks_analysis, letkf_analysis)


if __name__ == '__main__':
    unittest.main()
