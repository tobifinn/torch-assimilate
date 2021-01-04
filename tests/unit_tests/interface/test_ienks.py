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
from pytassim.interface.ienks import IEnKSTransform, IEnKSBundle
from pytassim.interface.etkf import ETKF
from pytassim.core.ienks import IEnKSTransformModule, IEnKSBundleModule
from pytassim.state import StateError
from pytassim.testing import dummy_obs_operator, if_gpu_decorator, \
    generate_random_weights


rnd = np.random.RandomState(42)

logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestIEnKSTransform(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator
        self.weights = generate_random_weights(len(self.state['ensemble']))
        self.algorithm = IEnKSTransform(lambda x: (self.state+1, self.state+2))

    def tearDown(self):
        self.state.close()
        self.obs.close()

    def test_tau_sets_core_module(self):
        old_id = id(self.algorithm._core_module)
        self.algorithm.tau = torch.tensor(0.67)
        self.assertNotEqual(id(self.algorithm._core_module), old_id)
        torch.testing.assert_allclose(self.algorithm.core_module.tau, 0.67)

    def test_tau_can_set_parameter(self):
        tau = torch.nn.Parameter(torch.tensor(0.67))
        self.algorithm.tau = tau
        self.assertEqual(self.algorithm.core_module.tau, tau)

    def test_float_tau_gets_converted_into_tensor(self):
        self.algorithm.tau = 0.67
        self.assertIsInstance(
            self.algorithm.core_module.tau, torch.Tensor
        )
        torch.testing.assert_allclose(
            self.algorithm.core_module.tau, 0.67
        )

    def test_tau_returns_core_module_tau(self):
        self.algorithm.core_module.tau = torch.tensor(0.01)
        self.assertEqual(self.algorithm.tau, self.algorithm.core_module.tau)

    def test_tau_is_bounded(self):
        with self.assertRaises(ValueError):
            self.algorithm.tau = -0.5

        with self.assertRaises(ValueError):
            self.algorithm.tau = 1.5

    def test_get_model_weights_returns_weights(self):
        returned_weights = self.algorithm.get_model_weights(self.weights)
        xr.testing.assert_identical(returned_weights, self.weights)

    def test_generate_prior_weights_returns_prior_weights(self):
        prior_weights = xr.DataArray(
            np.eye(10),
            coords={
                'ensemble': np.arange(10),
                'ensemble_new': np.arange(10)
            },
            dims=['ensemble', 'ensemble_new']
        )
        ret_weights = self.algorithm.generate_prior_weights(10)
        xr.testing.assert_identical(ret_weights, prior_weights)

    def test_propagate_model_applies_model_weights(self):
        self.algorithm.model = lambda x: (x+1, x)
        analysis_state = self.algorithm._apply_weights(self.state, self.weights)

        returned_state = self.algorithm._propagate_model(
            self.weights, self.state
        )
        xr.testing.assert_identical(returned_state, analysis_state)

    def test_propagate_model_propagates_model(self):
        self.algorithm.model = lambda x: (x+1, x+2)
        analysis_state = self.algorithm._apply_weights(self.state, self.weights)
        propagated_state = analysis_state + 2
        returned_state = self.algorithm._propagate_model(
            self.weights, self.state
        )
        xr.testing.assert_identical(returned_state, propagated_state)

    def test_propagate_model_tests_pseudo_state(self):
        self.algorithm.model = lambda x: (x+1, x.values)
        with self.assertRaises(TypeError):
            _ = self.algorithm._propagate_model(
                self.weights, self.state
            )

        self.algorithm.model = lambda x: (x+1, x[0])
        with self.assertRaises(StateError):
            _ = self.algorithm._propagate_model(
                self.weights, self.state
            )

    def test_estimate_weights_returns_right_weights(self):
        ens_obs = self.obs.obs.operator(self.obs, self.state)
        ens_mean = ens_obs.mean('ensemble')
        normed_ens_obs = self.obs.obs.mul_rcinv(ens_obs-ens_mean)
        normed_obs = self.obs.obs.mul_rcinv(self.obs['observations']-ens_mean)
        normed_ens_obs = normed_ens_obs.stack(obs_id=['time', 'obs_grid_1'])
        normed_obs = normed_obs.stack(obs_id=['time', 'obs_grid_1'])
        correct_weights = self.algorithm.module(
            self.weights.values, normed_ens_obs.values, normed_obs.values
        )
        correct_weights = self.weights.copy(data=correct_weights)
        returned_weights = self.algorithm.estimate_weights(
            self.state, self.weights, [self.obs], [ens_obs]
        )
        xr.testing.assert_identical(returned_weights, correct_weights)

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

    def test_ienks_with_linear_max_iter_1_equals_etkf(self):
        def linear_model(analysis):
            state = xr.concat([analysis,] * 3, dim='time')
            state['time'] = self.state['time'].values
            pseudo_state = state + 1
            return state, pseudo_state
        prior_state = self.state.isel(time=[0])
        propagated_state, pseudo_state = linear_model(prior_state)

        etkf = ETKF(inf_factor=1.0, smoother=True)
        etkf_analysis = etkf.assimilate(
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
        xr.testing.assert_allclose(ienks_analysis, etkf_analysis)


class TestIEnKSBundle(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator
        self.weights = generate_random_weights(len(self.state['ensemble']))
        self.algorithm = IEnKSBundle(lambda x: (self.state+1, self.state+2))

    def tearDown(self):
        self.state.close()
        self.obs.close()

    def test_tau_sets_core_module(self):
        old_id = id(self.algorithm._core_module)
        self.algorithm.tau = torch.tensor(0.67)
        self.assertNotEqual(id(self.algorithm._core_module), old_id)
        self.assertIsInstance(self.algorithm.core_module, IEnKSBundleModule)
        torch.testing.assert_allclose(self.algorithm.core_module.tau, 0.67)

    def test_tau_can_set_parameter(self):
        tau = torch.nn.Parameter(torch.tensor(0.67))
        self.algorithm.tau = tau
        self.assertEqual(self.algorithm.core_module.tau, tau)

    def test_float_tau_gets_converted_into_tensor(self):
        self.algorithm.tau = 0.67
        self.assertIsInstance(
            self.algorithm.core_module.tau, torch.Tensor
        )
        torch.testing.assert_allclose(
            self.algorithm.core_module.tau, 0.67
        )

    def test_tau_returns_core_module_tau(self):
        self.algorithm.core_module.tau = torch.tensor(0.01)
        self.assertEqual(self.algorithm.tau, self.algorithm.core_module.tau)

    def test_tau_sets_old_epsilon(self):
        old_eps = id(self.algorithm.core_module.epsilon)
        self.algorithm.core_module.tau = torch.tensor(0.01)
        self.assertEqual(old_eps, id(self.algorithm.core_module.epsilon))

    def test_epsilon_sets_core_module(self):
        old_id = id(self.algorithm._core_module)
        self.algorithm.epsilon = torch.tensor(0.67)
        self.assertNotEqual(id(self.algorithm._core_module), old_id)
        self.assertIsInstance(self.algorithm.core_module, IEnKSBundleModule)
        torch.testing.assert_allclose(self.algorithm.core_module.epsilon, 0.67)

    def test_epsilon_can_set_parameter(self):
        epsilon = torch.nn.Parameter(torch.tensor(0.67))
        self.algorithm.epsilon = epsilon
        self.assertEqual(self.algorithm.core_module.epsilon, epsilon)

    def test_float_epsilon_gets_converted_into_tensor(self):
        self.algorithm.epsilon = 0.67
        self.assertIsInstance(
            self.algorithm.core_module.epsilon, torch.Tensor
        )
        torch.testing.assert_allclose(
            self.algorithm.core_module.epsilon, 0.67
        )

    def test_epsilon_returns_core_module_epsilon(self):
        self.algorithm.core_module.epsilon = torch.tensor(0.01)
        self.assertEqual(self.algorithm.epsilon,
                         self.algorithm.core_module.epsilon)

    def test_epsilon_sets_old_tau(self):
        old_tau = id(self.algorithm.core_module.tau)
        self.algorithm.core_module.epsilon = torch.tensor(0.01)
        self.assertEqual(old_tau, id(self.algorithm.core_module.tau))

    def test_tau_is_bounded(self):
        with self.assertRaises(ValueError):
            self.algorithm.tau = -0.5

        with self.assertRaises(ValueError):
            self.algorithm.tau = 1.5

    def test_epsilon_is_bounded_from_below(self):
        self.algorithm.epsilon = 2.5
        with self.assertRaises(ValueError):
            self.algorithm.epsilon = -0.5
        torch.testing.assert_allclose(self.algorithm.epsilon, 2.5)

    def test_get_model_weights_returns_epsilon_weights(self):
        self.algorithm.epsilon = 1E-2
        weight_mean = self.weights.mean(dim='ensemble_new')
        prior_weights = self.algorithm.generate_prior_weights(
            len(weight_mean['ensemble'])
        )
        epsilon_weights = weight_mean + 1E-2 * prior_weights
        returned_weights = self.algorithm.get_model_weights(self.weights)
        xr.testing.assert_identical(returned_weights, epsilon_weights)

    def test_ienks_with_linear_max_iter_1_equals_etkf(self):
        def linear_model(analysis):
            state = xr.concat([analysis,] * 3, dim='time')
            state['time'] = self.state['time'].values
            pseudo_state = state + 1
            return state, pseudo_state
        prior_state = self.state.isel(time=[0])
        propagated_state, pseudo_state = linear_model(prior_state)

        etkf = ETKF(inf_factor=1.0, smoother=True)
        etkf_analysis = etkf.assimilate(
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
        xr.testing.assert_allclose(ienks_analysis, etkf_analysis)


if __name__ == '__main__':
    unittest.main()
