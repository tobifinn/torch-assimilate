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
import torch.autograd

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
        self.algorithm = IEnKSTransform(
            forward_model=lambda state, iter_num: (self.state+1, self.state+2)
        )

    def tearDown(self):
        self.state.close()
        self.obs.close()

    def test_tau_sets_core_module(self):
        old_id = id(self.algorithm._core_module)
        self.algorithm.tau = torch.tensor(0.67)
        self.assertNotEqual(id(self.algorithm._core_module), old_id)
        torch.testing.assert_allclose(self.algorithm.core_module.tau, 0.67)

    def test_precompute_weights_saves_weights_if_save_path(self):
        self.algorithm.weight_save_path = 'test.nc'
        dask_weights = self.weights.chunk({'ensemble': 1, 'ensemble_new': 1})

        with patch(
                'xarray.core.dataarray.Dataset.to_netcdf',
                return_value=None
        ) as save_patch, patch(
                    'xarray.open_dataarray', return_value=dask_weights
        ) as load_patch:
            returned_weights = self.algorithm.precompute_weights(dask_weights)
        save_patch.assert_called_once_with('test.nc', compute=True)
        load_patch.assert_called_once_with(
            'test.nc', chunks=None
        )
        xr.testing.assert_identical(returned_weights, dask_weights)

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

    def test_innerloop_returns_right_weights(self):
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
        returned_weights = self.algorithm.inner_loop(
            self.state, self.weights, [self.obs], [ens_obs]
        )
        xr.testing.assert_identical(returned_weights, correct_weights)

    def test_algorithm_skips_first_propagation(self):
        self.algorithm.max_iter = 1
        self.algorithm.model = MagicMock()
        self.algorithm.model.return_value = (self.state, self.state+1)
        _ = self.algorithm.assimilate(
            self.state, self.obs, self.state,
            analysis_time=self.state.time[-1].values
        )
        self.algorithm.model.assert_not_called()

    def test_algorithm_uses_later_propagations(self):
        self.algorithm.max_iter = 2
        self.algorithm.forward_model = MagicMock()
        self.algorithm.forward_model.return_value = (self.state, self.state+1)
        _ = self.algorithm.assimilate(
            self.state, self.obs, self.state,
            analysis_time=self.state.time[-1].values
        )
        self.algorithm.forward_model.assert_called_once()

    def test_algorithm_works(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      None, ana_time)
        self.assertFalse(np.any(np.isnan(assimilated_state.values)))

    def test_algorithm_works_shited_ens(self):
        self.state['ensemble'] = np.arange(1, len(self.state['ensemble'])+1)
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
        def linear_model(state, iter_num):
            state = xr.concat([state,] * 3, dim='time')
            state['time'] = self.state['time'].values
            pseudo_state = state + 1
            return state, pseudo_state
        prior_state = self.state.isel(time=[0])
        propagated_state, pseudo_state = linear_model(prior_state, 0)

        etkf = ETKF(inf_factor=1.0, smoother=True)
        etkf_analysis = etkf.assimilate(
            propagated_state, self.obs, pseudo_state,
            analysis_time=prior_state.time.values
        )

        self.algorithm.max_iter = 1
        self.algorithm.tau = 1.0
        self.algorithm.forward_model = linear_model
        self.algorithm.smoother = True
        ienks_analysis = self.algorithm.assimilate(
            prior_state, self.obs, analysis_time=prior_state.time.values
        )
        xr.testing.assert_allclose(ienks_analysis, etkf_analysis)

    def test_chunks_return_None(self):
        self.assertIsNone(self.algorithm.chunks)


class TestIEnKSBundle(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator
        self.weights = generate_random_weights(len(self.state['ensemble']))
        self.algorithm = IEnKSBundle(
            forward_model=lambda state, iter_num: (self.state+1, self.state+2)
        )
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
            weight_mean['ensemble'].values
        )
        epsilon_weights = weight_mean + 1E-2 * prior_weights
        returned_weights = self.algorithm._get_model_weights(self.weights)
        xr.testing.assert_identical(returned_weights, epsilon_weights)

    def test_get_model_weights_called_by_propagate_model(self):
        with patch('pytassim.interface.ienks.IEnKSBundle._get_model_weights',
                   return_value=self.weights) as model_weights_patch:
            self.algorithm.propagate_model(
                self.weights, self.state, iter_num=0
            )
        model_weights_patch.assert_called_once_with(self.weights)

    def test_propagate_model_returns_same_as_boc14_algo_2_line_6(self):
        curr_state = self.state.isel(time=[0])
        ens_perts = curr_state-curr_state.mean('ensemble')
        epsilon_ensemble = curr_state.mean('ensemble') + 0.01 * ens_perts
        epsilon_ensemble = epsilon_ensemble.transpose(*curr_state.dims)
        self.algorithm.forward_model = lambda x, iter_num: (x, x)
        self.algorithm.epsilon = 0.01
        weights = self.algorithm.generate_prior_weights(
            ens_perts['ensemble'].values
        )
        ret_ensemble = self.algorithm.propagate_model(weights, curr_state)
        xr.testing.assert_allclose(
            epsilon_ensemble, ret_ensemble,
            rtol=1E-10, atol=1E-12
        )

    def test_gradient_test_bundle(self):
        def quadratic_model(state, iter_num):
            return state, (state*0.5)**2
        self.algorithm.epsilon = 1E-7
        self.algorithm.forward_model = quadratic_model
        curr_state = self.state.isel(var_name=[0], time=[0])
        pseudo_state = self.algorithm.propagate_model(self.weights, curr_state)
        pseudo_state = pseudo_state-pseudo_state.mean('ensemble')
        torch_pseudo = torch.from_numpy(pseudo_state.values).view(10, 40)
        torch_weights = torch.from_numpy(self.weights.values)
        _, w_perts_inv, _ = self.algorithm.core_module._decompose_weights(
            torch_weights, 10
        )
        bundle_dh_dw = self.algorithm.core_module._get_dh_dw(
            torch_pseudo, w_perts_inv
        )

        torch_weights_mean = torch.nn.Parameter(
            torch_weights.mean(dim=1, keepdim=True),
            requires_grad=True
        )

        torch_state = torch.from_numpy(curr_state.values).view(10, 40)
        torch_state_mean = torch_state.mean(dim=0, keepdim=True)
        torch_state_perts = torch_state - torch_state_mean
        torch_analysis = torch_state_mean + torch.einsum(
            'ig,ij->jg', torch_state_perts, torch_weights_mean
        )
        torch_propagated = quadratic_model(torch_analysis, 0)[1][0]
        torch_dh_dw = torch.cat([
            torch.autograd.grad(
                grid_point, torch_weights_mean,
                retain_graph=True
            )[0]
            for grid_point in torch_propagated
        ], dim=-1)

        torch.testing.assert_allclose(
            bundle_dh_dw, torch_dh_dw,
            rtol=1E-5, atol=1E-7
        )

    def test_ienks_with_linear_max_iter_1_equals_etkf(self):
        def linear_model(state, iter_num):
            state = xr.concat([state,] * 3, dim='time')
            state['time'] = self.state['time'].values
            pseudo_state = state + 1
            return state, pseudo_state
        prior_state = self.state.isel(time=[0])
        propagated_state, pseudo_state = linear_model(prior_state, 0)

        etkf = ETKF(inf_factor=1.0, smoother=True)
        etkf_analysis = etkf.assimilate(
            propagated_state, self.obs, pseudo_state,
            analysis_time=prior_state.time.values
        )

        self.algorithm.max_iter = 1
        self.algorithm.tau = 1.0
        self.algorithm.epsilon = 1.0
        self.algorithm.forward_model = linear_model
        self.algorithm.smoother = True
        ienks_analysis = self.algorithm.assimilate(
            prior_state, self.obs, analysis_time=prior_state.time.values
        )
        xr.testing.assert_allclose(ienks_analysis, etkf_analysis)


if __name__ == '__main__':
    unittest.main()
