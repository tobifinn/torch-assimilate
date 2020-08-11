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
import torch
import torch.jit
import torch.nn
import scipy.linalg
import scipy.linalg.blas

# Internal modules
import pytassim.state
import pytassim.observation
from pytassim.assimilation.filter.etkf import ETKFCorr, ETKFUncorr
from pytassim.assimilation.filter.etkf_module import ETKFWeightsModule
from pytassim.testing import dummy_obs_operator, if_gpu_decorator
from pytassim.assimilation.filter import etkf_module


logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestETKFModule(unittest.TestCase):
    def setUp(self):
        self.module = ETKFWeightsModule()
        self.state, self.obs = self._create_matrices()
        innov = (self.obs['observations']-self.state.mean('ensemble'))
        innov = innov.values.reshape(-1)
        hx_perts = self.state.values.reshape(2, 1)
        obs_cov = self.obs['covariance'].values
        prepared_states = [innov, hx_perts, obs_cov]
        torch_states = [torch.from_numpy(s).float() for s in prepared_states]
        innov, hx_perts, obs_cov = torch_states
        obs_cinv = torch.cholesky(obs_cov).inverse()
        self.normed_perts = hx_perts @ obs_cinv
        self.normed_obs = (innov @ obs_cinv).view(1, 1)

    def _create_matrices(self):
        ens_obs = np.array([0.5, -0.5])
        obs = np.array([0.2, ])
        obs_var = np.array([0.5, ])
        grid = np.array([0, ])
        time = np.array([0, ])
        var_name = np.array([0, ])
        ensemble = np.arange(2)
        state = xr.DataArray(
            ens_obs.reshape(1, 1, 2, 1),
            coords=dict(
                time=time,
                var_name=var_name,
                ensemble=ensemble,
                grid=grid
            ),
            dims=('var_name', 'time', 'ensemble', 'grid')
        )
        obs_da = xr.DataArray(
            obs.reshape(1, 1),
            coords=dict(
                time=time,
                obs_grid_1=grid
            ),
            dims=('time', 'obs_grid_1')
        )
        obs_cov_da = xr.DataArray(
            obs_var.reshape(1, 1),
            coords=dict(
                obs_grid_1=grid,
                obs_grid_2=grid
            ),
            dims=('obs_grid_1', 'obs_grid_2')
        )
        obs_ds = xr.Dataset(
            {
                'observations': obs_da,
                'covariance': obs_cov_da
            }
        )
        return state, obs_ds

    def test_is_module(self):
        self.assertIsInstance(self.module, torch.nn.Module)
        try:
            _ = torch.jit.script(self.module)
        except RuntimeError:
            raise AssertionError('JIT is not possible!')

    def test_inf_factor_float_to_tensor(self):
        self.module._inf_factor = None
        self.assertIsNone(self.module._inf_factor)
        self.module.inf_factor = 1.2
        self.assertIsInstance(self.module._inf_factor, torch.Tensor)
        torch.testing.assert_allclose(
            self.module._inf_factor, torch.tensor(1.2)
        )

    def test_inf_factor_uses_tensor(self):
        self.module._inf_factor = None
        self.assertIsNone(self.module._inf_factor)
        test_tensor = torch.tensor(1.2)
        self.module.inf_factor = test_tensor
        self.assertEqual(id(self.module._inf_factor), id(test_tensor))

    def test_dot_product(self):
        right_dot_product = self.normed_perts @ self.normed_perts.t()
        out_dot = etkf_module._dot_product(self.normed_perts, self.normed_perts)
        torch.testing.assert_allclose(out_dot, right_dot_product)

    def test_right_cov(self):
        ret_kernel = etkf_module._dot_product(self.normed_perts,
                                              self.normed_perts)
        ret_evd = etkf_module._evd(ret_kernel, 1)
        evals, evects, evals_inv, evects_inv = ret_evd

        cov_analysed = torch.matmul(evects, torch.diagflat(evals_inv))
        cov_analysed = torch.matmul(cov_analysed, evects_inv)

        right_cov = np.array([
            [0.75, 0.25],
            [0.25, 0.75]
        ])

        np.testing.assert_array_almost_equal(cov_analysed, right_cov)

    def test_rev_evd(self):
        ret_kernel = etkf_module._dot_product(self.normed_perts,
                                              self.normed_perts)
        evals, evects, evals_inv, evects_inv = etkf_module._evd(
            ret_kernel, 1
        )
        right_rev = torch.mm(evects, torch.diagflat(evals))
        right_rev = torch.mm(right_rev, evects_inv)

        ret_rev = etkf_module._rev_evd(evals, evects, evects_inv)
        torch.testing.assert_allclose(ret_rev, right_rev)

    def test_right_w_eigendecomposition(self):
        ret_prec = self.normed_perts @ self.normed_perts.t()
        evals, evects = np.linalg.eigh(ret_prec)
        evals = evals + 1
        evals_inv_sqrt = np.diagflat(np.sqrt(1/evals))
        w_pert = np.dot(evals_inv_sqrt, evects.T)
        w_pert = np.dot(evects, w_pert)

        ret_perts = etkf_module.gen_weights(self.normed_perts, self.normed_obs,
                                            1)[2]
        np.testing.assert_array_almost_equal(ret_perts.numpy(), w_pert)

    def test_returns_w_mean(self):
        correct_gain = np.array([0.5, -0.5])
        correct_wa = (correct_gain * 0.2).reshape(2, 1)
        ret_wa = etkf_module.gen_weights(self.normed_perts, self.normed_obs, 1)[1]
        np.testing.assert_array_almost_equal(ret_wa.numpy(), correct_wa)

    def test_returns_w_perts(self):
        right_cov = np.array([
            [0.75, 0.25],
            [0.25, 0.75]
        ])
        return_perts = etkf_module.gen_weights(self.normed_perts, self.normed_obs,
                                               1)[2]
        return_perts = return_perts.numpy()
        ret_cov = np.matmul(return_perts, return_perts.T)
        np.testing.assert_array_almost_equal(ret_cov, right_cov)

    def test_returns_w_cov(self):
        right_cov = np.array([
            [0.75, 0.25],
            [0.25, 0.75]
        ])
        return_cov = etkf_module.gen_weights(self.normed_perts, self.normed_obs,
                                             1)[3]
        return_cov = return_cov.numpy()
        np.testing.assert_array_almost_equal(return_cov, right_cov)

    def test_returns_weights(self):
        weights, w_mean, w_perts, _ = etkf_module.gen_weights(
            self.normed_perts, self.normed_obs, 1
        )
        torch.testing.assert_allclose(weights, w_mean+w_perts)

    def test_apply_weights_ens_mean(self):
        weights, w_mean, _, _ = etkf_module.gen_weights(
            self.normed_perts, self.normed_obs, 1
        )
        state_mean, state_perts = self.state.state.split_mean_perts()
        perts_t = state_perts.transpose('var_name', 'time', 'grid', 'ensemble')
        del_ana_mean = np.matmul(perts_t.values, w_mean.numpy().squeeze())
        ana_mean = state_mean + del_ana_mean
        ret_state = self.algorithm._apply_weights(weights, state_mean,
                                                  state_perts)
        np.testing.assert_almost_equal(ret_state.mean('ensemble').values,
                                       ana_mean.values)

    def test_apply_weights_perts(self):
        weights, _, w_perts, _ = etkf_module.gen_weights(
            self.normed_perts, self.normed_obs, 1
        )
        state_mean, state_perts = self.state.state.split_mean_perts()
        perts_t = state_perts.transpose('var_name', 'time', 'grid', 'ensemble')
        del_ana_perts = np.matmul(perts_t.values, w_perts.numpy())
        ret_state = self.algorithm._apply_weights(weights, state_mean,
                                                  state_perts)
        ret_ana_perts = ret_state - ret_state.mean('ensemble')
        np.testing.assert_equal(ret_ana_perts, del_ana_perts)


class TestETKFCorr(unittest.TestCase):
    def setUp(self):
        self.algorithm = ETKFCorr()
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator

    def tearDown(self):
        self.state.close()
        self.obs.close()

    def test_prepare_obs_stackes_and_concats_obs(self):
        obs_stacked = self.obs['observations'].stack(
            obs_id=('time', 'obs_grid_1')
        )
        obs_concat = xr.concat((obs_stacked, obs_stacked), dim='obs_id').values
        returned_obs, _, _ = self.algorithm._prepare_obs(
            (self.obs, self.obs)
        )
        np.testing.assert_equal(obs_concat, returned_obs)

    def test_prepare_obs_returns_obs_grid(self):
        obs_stacked = self.obs['observations'].stack(
            obs_id=('time', 'obs_grid_1')
        )
        obs_grid = np.tile(obs_stacked.obs_grid_1.values, 2)
        _, _, returned_grid = self.algorithm._prepare_obs(
            (self.obs, self.obs)
        )
        np.testing.assert_equal(obs_grid.reshape(-1, 1), returned_grid)

    def test_prepare_obs_returns_obs_cov_matrix(self):
        len_time = len(self.obs.time)
        stacked_cov = [self.obs['covariance'].values] * len_time
        stacked_cov = scipy.linalg.block_diag(*stacked_cov)
        block_diag = scipy.linalg.block_diag(stacked_cov, stacked_cov)
        _, returned_cov, _ = self.algorithm._prepare_obs(
            (self.obs, self.obs)
        )
        np.testing.assert_equal(returned_cov, block_diag)

    def test_prepare_obs_works_for_single_obs(self):
        obs_stacked = self.obs['observations'].stack(
            obs_id=('time', 'obs_grid_1')
        )
        len_time = len(self.obs.time)
        stacked_cov = [self.obs['covariance'].values] * len_time
        stacked_cov = scipy.linalg.block_diag(*stacked_cov)
        returned_obs, returned_cov, returned_grid = self.algorithm._prepare_obs(
            (self.obs, )
        )
        np.testing.assert_equal(returned_obs, obs_stacked.values)
        np.testing.assert_equal(returned_cov, stacked_cov)
        np.testing.assert_equal(returned_grid, obs_stacked.obs_grid_1.values.reshape(-1, 1))

    def test_prepare_state_returns_state_array(self):
        hx = self.obs.obs.operator(self.state)
        hx_stacked = hx.stack(obs_id=('time', 'obs_grid_1'))
        hx_concat = xr.concat([hx_stacked, hx_stacked], dim='obs_id')
        hx_mean, hx_pert = hx_concat.state.split_mean_perts()
        returned_mean, returned_pert, _ = self.algorithm._prepare_back_obs(
            self.state, (self.obs, self.obs)
        )
        np.testing.assert_equal(returned_mean, hx_mean.values)
        np.testing.assert_equal(returned_pert, hx_pert.T.values)

    def test_prepare_state_returns_filtered_obs(self):
        obs_list = (self.obs, self.obs.copy())
        _, _, returned_obs = self.algorithm._prepare_back_obs(
            self.state, obs_list
        )
        self.assertEqual(len(returned_obs), 1)
        self.assertEqual(id(self.obs), returned_obs[0])

    def test_prepare_calls_prepare_state(self):
        obs_tuple = (self.obs, self.obs.copy())
        prepared_state = self.algorithm._prepare_back_obs(self.state, obs_tuple)
        trg = 'pytassim.assimilation.filter.etkf.ETKFCorr._prepare_back_obs'
        with patch(trg, return_value=prepared_state) as prepare_patch:
            _ = self.algorithm._prepare(self.state, obs_tuple)
        prepare_patch.assert_called_once_with(self.state, obs_tuple)

    def test_prepare_calls_prepare_obs_with_filtered_obs(self):
        obs_tuple = (self.obs, self.obs.copy())
        prepared_obs = self.algorithm._prepare_obs((self.obs, ))
        with patch('pytassim.assimilation.filter.etkf.ETKFCorr._prepare_obs',
                   return_value=prepared_obs) as prepare_patch:
            _ = self.algorithm._prepare(self.state, obs_tuple)
        prepare_patch.assert_called_once()
        self.assertListEqual([self.obs, ], prepare_patch.call_args[0][0])

    def test_prepare_returns_necessary_variables(self):
        obs_tuple = (self.obs, self.obs.copy())
        prepared_state = self.algorithm._prepare_back_obs(self.state, obs_tuple)
        prepared_obs = self.algorithm._prepare_obs((self.obs, ))
        innov = prepared_obs[0] - prepared_state[0]

        returned_state = self.algorithm._prepare(self.state, obs_tuple)
        np.testing.assert_equal(innov, returned_state[0])
        np.testing.assert_equal(prepared_state[1], returned_state[1])
        np.testing.assert_equal(prepared_obs[1], returned_state[2])
        np.testing.assert_equal(prepared_obs[2], returned_state[3])

    def test_update_calls_prepare_with_pseudo_state(self):
        pseudo_state = self.state + 1
        obs_tuple = (self.obs, self.obs.copy())
        returned_state = self.algorithm._prepare(pseudo_state, obs_tuple)
        with patch('pytassim.assimilation.filter.etkf.ETKFCorr._prepare',
                   return_value=returned_state) as prepare_patch:
            self.algorithm.update_state(self.state, obs_tuple, pseudo_state,
                                        self.state.time[-1].values)
        prepare_patch.assert_called_once_with(pseudo_state, obs_tuple)

    def test_update_state_uses_prepare_function(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        with patch('pytassim.assimilation.filter.etkf.ETKFCorr._prepare',
                   return_value=prepared_states) as prepare_patch:
            _ = self.algorithm.update_state(
                self.state, obs_tuple, self.state, self.state.time[-1].values
            )
        prepare_patch.assert_called_once_with(self.state, obs_tuple)

    def test_get_obs_cinvUses_choleksy_inv(self):
        perts = torch.zeros(100, 5).normal_()
        cov = (perts.t() @ perts) / 99
        chol_decomp = np.linalg.cholesky(cov.numpy())
        right_cinv = np.linalg.inv(chol_decomp)
        ret_cinv = etkf_module.estimate_cinv(cov)
        np.testing.assert_almost_equal(ret_cinv, right_cinv)

    def test_transfer_states_transfers_arguments_to_tensor(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        ret_states = self.algorithm._states_to_torch(*prepared_states)
        for k, state in enumerate(ret_states):
            self.assertIsInstance(state, torch.Tensor)
            np.testing.assert_array_equal(state.numpy(), prepared_states[k])

    @if_gpu_decorator
    def test_transfer_states_transfers_to_gpus(self):
        self.algorithm.gpu = True
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        ret_states = self.algorithm._states_to_torch(*prepared_states)
        for k, state in enumerate(ret_states):
            self.assertIsInstance(state, torch.Tensor)
            self.assertTrue(state.is_cuda)
            np.testing.assert_array_equal(state.cpu().numpy(),
                                          prepared_states[k])

    def test_update_states_uses_states_to_torch(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        torch_states = self.algorithm._states_to_torch(*prepared_states)[:-1]
        trg = 'pytassim.assimilation.filter.etkf.ETKFCorr._states_to_torch'
        with patch(trg, return_value=torch_states) as torch_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple, self.state,
                                            ana_time)
        torch_patch.assert_called_once()

    def test_algorithm_works(self):
        self.algorithm.inf_factor = 1.2
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      None, ana_time)
        self.assertFalse(np.any(np.isnan(assimilated_state.values)))


class TestETKFUncorr(unittest.TestCase):
    def setUp(self):
        self.algorithm = ETKFUncorr()
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        self.back_prec = self.algorithm._get_back_prec(
            len(self.state.ensemble))
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs['covariance'] = xr.DataArray(
            np.diag(self.obs.covariance.values),
            coords={
                'obs_grid_1': self.obs.obs_grid_1
            },
            dims=['obs_grid_1']
        )
        self.obs.obs.operator = dummy_obs_operator

    def tearDown(self):
        self.state.close()
        self.obs.close()
    
    def test_etkf_sets_gen_func_to_uncorr(self):
        self.assertEqual(self.algorithm._gen_weights_func,
                         etkf_module.gen_weights_uncorr)

    def test_etkf_sets_correlated_to_false(self):
        self.assertFalse(self.algorithm._correlated)

    def test_prepare_obs_returns_diagonal_elements_of_cov(self):
        len_time = len(self.obs.time)
        stacked_cov = np.concatenate(
            [self.obs['covariance'].values] * len_time
        )
        diag_cov = np.concatenate([stacked_cov, stacked_cov])
        _, returned_cov, _ = self.algorithm._prepare_obs(
            (self.obs, self.obs)
        )
        np.testing.assert_equal(returned_cov, diag_cov)

    def test_calculate_c_returns_same_for_diag(self):
        obs_tuple = [self.obs, ] * 5
        _, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        _, hx_perts, _ = self.algorithm._prepare_back_obs(self.state, obs_tuple)
        obs_cov = torch.tensor(obs_cov)
        hx_perts = torch.tensor(hx_perts)
        diag_c = etkf_module._compute_c_diag(hx_perts, obs_cov)
        chol_c = etkf_module._compute_c_chol(hx_perts, torch.diag(obs_cov))
        torch.testing.assert_allclose(diag_c, chol_c)

    def test_diagonal_inverse_returns_inverse_of_diagonal_matrix(self):
        obs_tuple = [self.obs, ] * 5
        _, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        _, hx_perts, _ = self.algorithm._prepare_back_obs(self.state, obs_tuple)
        est_c = np.matmul(hx_perts.T, np.linalg.inv(np.diag(obs_cov)))
        self.assertTupleEqual(est_c.shape, hx_perts.T.shape)

        t_obs_cov = torch.tensor(obs_cov)
        t_hx_perts = torch.tensor(hx_perts)
        ret_c = etkf_module._compute_c_diag(
            t_hx_perts, t_obs_cov
        ).numpy()
        self.assertTupleEqual(ret_c.shape, hx_perts.T.shape)

    def test_algorithm_works(self):
        self.algorithm.inf_factor = 1.2
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      None, ana_time)
        self.assertFalse(np.any(np.isnan(assimilated_state.values)))


if __name__ == '__main__':
    unittest.main()
