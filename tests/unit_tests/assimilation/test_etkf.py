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
from pytassim.assimilation.filter.etkf import ETKFCorr, ETKFUncorr
from pytassim.testing import dummy_obs_operator, if_gpu_decorator


logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


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

    def test_inf_factor_sets_analyser(self):
        old_id = id(self.algorithm._analyser)
        self.algorithm.inf_factor = 3.2
        self.assertNotEqual(id(self.algorithm._analyser), old_id)
        self.assertEqual(self.algorithm._analyser.inf_factor, 3.2)

    def test_prepare_obs_stackes_and_concats_obs(self):
        obs_stacked = self.obs['observations'].stack(
            obs_id=('time', 'obs_grid_1')
        )
        obs_concat = xr.concat((obs_stacked, obs_stacked), dim='obs_id').values
        returned_obs, _ = self.algorithm._prepare_obs(
            (self.obs, self.obs)
        )
        np.testing.assert_equal(obs_concat, returned_obs)

    def test_prepare_obs_returns_obs_grid(self):
        obs_stacked = self.obs['observations'].stack(
            obs_id=('time', 'obs_grid_1')
        )
        obs_grid = np.tile(obs_stacked.obs_grid_1.values, 2)
        _, returned_grid = self.algorithm._prepare_obs(
            (self.obs, self.obs)
        )
        np.testing.assert_equal(obs_grid.reshape(-1, 1), returned_grid)

    def test_prepare_obs_returns_obs_grid_multiindex(self):
        multiindex_grid = pd.MultiIndex.from_product(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.1, 0.2, 0.3, 0.4]]
        )
        self.obs['obs_grid_1'] = self.obs['obs_grid_2'] = multiindex_grid
        obs_grid = np.tile(multiindex_grid.to_frame().values, (6, 1))
        _, returned_grid = self.algorithm._prepare_obs(
            (self.obs, self.obs)
        )
        np.testing.assert_equal(obs_grid, returned_grid)

    def test_prepare_obs_returns_obs_cov_matrix(self):
        len_time = len(self.obs.time)
        stacked_cov = [self.obs['covariance'].values] * len_time
        stacked_cov = scipy.linalg.block_diag(*stacked_cov)
        block_diag = scipy.linalg.block_diag(stacked_cov, stacked_cov)
        returned_cov = self.algorithm._get_obs_cov(
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
        returned_cov = self.algorithm._get_obs_cov((self.obs, ))
        returned_obs, returned_grid = self.algorithm._prepare_obs(
            (self.obs, )
        )
        np.testing.assert_equal(returned_obs, obs_stacked.values)
        np.testing.assert_equal(returned_grid,
                                obs_stacked.obs_grid_1.values.reshape(-1, 1))
        np.testing.assert_equal(returned_cov, stacked_cov)

    def test_prepare_state_returns_state_array(self):
        hx = self.obs.obs.operator(self.obs, self.state)
        hx_stacked = hx.stack(obs_id=('time', 'obs_grid_1'))
        hx_concat = xr.concat([hx_stacked, hx_stacked], dim='obs_id')
        pseudo_obs, _ = self.algorithm._get_pseudo_obs(
            self.state, (self.obs, self.obs)
        )
        np.testing.assert_equal(pseudo_obs, hx_concat.data)

    def test_prepare_state_returns_filtered_obs(self):
        obs_list = (self.obs, self.obs.copy())
        _, returned_obs = self.algorithm._get_pseudo_obs(
            self.state, obs_list
        )
        self.assertEqual(len(returned_obs), 1)
        self.assertEqual(id(self.obs), returned_obs[0])

    def test_prepare_calls_prepare_state(self):
        obs_tuple = (self.obs, self.obs.copy())
        prepared_state = self.algorithm._get_pseudo_obs(self.state, obs_tuple)
        trg = 'pytassim.assimilation.filter.etkf.ETKFCorr._get_pseudo_obs'
        with patch(trg, return_value=prepared_state) as prepare_patch:
            _ = self.algorithm._get_states(self.state, obs_tuple)
        prepare_patch.assert_called_once_with(self.state, obs_tuple)

    def test_prepare_calls_prepare_obs_with_filtered_obs(self):
        obs_tuple = (self.obs, self.obs.copy())
        prepared_obs = self.algorithm._prepare_obs((self.obs, ))
        with patch('pytassim.assimilation.filter.etkf.ETKFCorr._prepare_obs',
                   return_value=prepared_obs) as prepare_patch:
            _ = self.algorithm._get_states(self.state, obs_tuple)
        prepare_patch.assert_called_once()
        self.assertListEqual([self.obs, ], prepare_patch.call_args[0][0])

    def test_cat_pseudo_obs_returns_numpy(self):
        obs_tuple = (self.obs, self.obs.copy())
        pseudo_obs_list, _ = self.algorithm._apply_obs_operator(self.state,
                                                                obs_tuple)
        pseudo_obs = self.algorithm._cat_pseudo_obs(pseudo_obs_list)
        self.assertIsInstance(pseudo_obs, np.ndarray)

    def test_prepare_returns_necessary_variables(self):
        obs_tuple = (self.obs, self.obs.copy())
        prepared_state, filtered_obs = self.algorithm._get_pseudo_obs(
            self.state, obs_tuple
        )
        obs_cov = self.algorithm._get_obs_cov(filtered_obs)
        prepared_obs = self.algorithm._prepare_obs(filtered_obs)

        returned_state = self.algorithm._get_states(self.state, obs_tuple)
        np.testing.assert_equal(returned_state[0], prepared_state)
        np.testing.assert_equal(returned_state[1], prepared_obs[0])
        np.testing.assert_equal(returned_state[2], obs_cov)
        np.testing.assert_equal(returned_state[3], prepared_obs[1])

    def test_update_calls_prepare_with_pseudo_state(self):
        pseudo_state = self.state + 1
        obs_tuple = (self.obs, self.obs.copy())
        returned_state = self.algorithm._get_states(pseudo_state, obs_tuple)
        with patch('pytassim.assimilation.filter.etkf.ETKFCorr._get_states',
                   return_value=returned_state) as prepare_patch:
            self.algorithm.update_state(self.state, obs_tuple, pseudo_state,
                                        self.state.time[-1].values)
        prepare_patch.assert_called_once_with(pseudo_state, obs_tuple)

    def test_update_state_uses_prepare_function(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._get_states(self.state, obs_tuple)
        with patch('pytassim.assimilation.filter.etkf.ETKFCorr._get_states',
                   return_value=prepared_states) as prepare_patch:
            _ = self.algorithm.update_state(
                self.state, obs_tuple, self.state, self.state.time[-1].values
            )
        prepare_patch.assert_called_once_with(self.state, obs_tuple)

    def test_transfer_states_transfers_arguments_to_tensor(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._get_states(self.state, obs_tuple)
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

    def test_get_obs_cinv(self):
        perts = torch.zeros(100, 5).normal_()
        cov = (perts.t() @ perts) / 99
        chol_decomp = np.linalg.cholesky(cov.numpy())
        right_cinv = np.linalg.inv(chol_decomp)
        ret_cinv = self.algorithm._get_chol_inverse(cov)
        np.testing.assert_almost_equal(ret_cinv, right_cinv)

    def test_uses_get_obs_cinv(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        prepared_states = self.algorithm._get_states(self.state, obs_tuple)
        torch_states = self.algorithm._states_to_torch(*prepared_states)[:-1]
        obs_cinv = self.algorithm._get_chol_inverse(torch_states[-1])

        trg = 'pytassim.assimilation.filter.etkf.ETKFCorr._get_chol_inverse'
        with patch(trg, return_value=obs_cinv) as cinv_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple, self.state,
                                            ana_time)
        cinv_patch.assert_called_once()

    def test_normalise_centers_pseudo_obs(self):
        test_pseudo = torch.zeros(10, 5).normal_(mean=1.)
        test_obs = torch.zeros(1, 5).normal_()
        test_mean = test_pseudo.mean(dim=-2, keepdim=True)
        perts = torch.zeros(100, 5).normal_()
        cov = (perts.t() @ perts) / 99
        chol_decomp = np.linalg.cholesky(cov.numpy())
        test_cinv = np.linalg.inv(chol_decomp)
        test_cinv = torch.from_numpy(test_cinv).float()

        norm_perts, _ = self.algorithm._normalise_obs(test_pseudo, test_obs,
                                                      test_cinv)

        centered_tensor = test_pseudo-test_mean
        ret_tensor = norm_perts.numpy() @ chol_decomp
        torch.testing.assert_allclose(ret_tensor, centered_tensor)

    def test_normalise_centers_obs(self):
        test_pseudo = torch.zeros(10, 5).normal_(mean=1.)
        test_obs = torch.zeros(1, 5).normal_()
        test_mean = test_pseudo.mean(dim=-2, keepdim=True)
        perts = torch.zeros(100, 5).normal_()
        cov = (perts.t() @ perts) / 99
        chol_decomp = np.linalg.cholesky(cov.numpy())
        test_cinv = np.linalg.inv(chol_decomp)
        test_cinv = torch.from_numpy(test_cinv).float()

        _, normed_obs = self.algorithm._normalise_obs(test_pseudo, test_obs,
                                                      test_cinv)

        centered_tensor = test_obs-test_mean
        ret_tensor = normed_obs.numpy() @ chol_decomp
        torch.testing.assert_allclose(ret_tensor, centered_tensor)

    def test_normalise_returns_normed_perts_obs(self):
        test_pseudo = torch.zeros(10, 5).normal_(mean=1.)
        test_obs = torch.zeros(1, 5).normal_()
        test_mean = test_pseudo.mean(dim=-2, keepdim=True)
        perts = torch.zeros(100, 5).normal_()
        cov = (perts.t() @ perts) / 99
        chol_decomp = np.linalg.cholesky(cov.numpy())
        test_cinv = np.linalg.inv(chol_decomp)
        test_cinv = torch.from_numpy(test_cinv).float()

        normed_perts = (test_pseudo-test_mean) @ test_cinv
        normed_obs = (test_obs-test_mean)  @ test_cinv

        ret_perts, ret_obs = self.algorithm._normalise_obs(
            test_pseudo, test_obs, test_cinv
        )

        torch.testing.assert_allclose(ret_perts, normed_perts)
        torch.testing.assert_allclose(ret_obs, normed_obs)

    def test_normalise_uses_multiply_cinv(self):
        test_pseudo = torch.zeros(10, 5).normal_(mean=1.)
        test_obs = torch.zeros(1, 5).normal_()
        test_mean = test_pseudo.mean(dim=-2, keepdim=True)
        perts = torch.zeros(100, 5).normal_()
        cov = (perts.t() @ perts) / 99
        chol_decomp = np.linalg.cholesky(cov.numpy())
        test_cinv = np.linalg.inv(chol_decomp)
        test_cinv = torch.from_numpy(test_cinv).float()

        trg = 'pytassim.assimilation.filter.etkf.ETKFCorr._mul_cinv'
        with patch(trg, return_value=test_pseudo) as cinv_patch:
            _ = self.algorithm._normalise_obs(
                test_pseudo, test_obs, test_cinv
            )
        cinv_patch.assert_called()
        self.assertEqual(cinv_patch.call_count, 2)

    def test_uses_center_tensors(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        prepared_states = self.algorithm._get_states(self.state, obs_tuple)
        torch_states = self.algorithm._states_to_torch(*prepared_states)[:-1]
        obs_cinv = self.algorithm._get_chol_inverse(torch_states[-1])
        normed_tensors = self.algorithm._normalise_obs(*torch_states[:2],
                                                       obs_cinv)

        trg = 'pytassim.assimilation.filter.etkf.ETKFCorr._normalise_obs'
        with patch(trg, return_value=normed_tensors) as center_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple, self.state,
                                            ana_time)
        center_patch.assert_called_once()

    def test_algorithm_works(self):
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

    def test_etkf_sets_correlated_to_false(self):
        self.assertFalse(self.algorithm._correlated)

    def test_normalise_cinv_works_with_inv_sqrt(self):
        state = torch.zeros(10, 5).normal_()
        sqrt_inv = torch.zeros(5).uniform_(1, 5)
        norm_state = torch.mm(state, torch.eye(5) * sqrt_inv)
        ret_state = self.algorithm._mul_cinv(state, sqrt_inv)
        torch.testing.assert_allclose(ret_state, norm_state)

    def test_get_chol_inverse_ret_sqrt_inv(self):
        cov = torch.zeros(5).uniform_(1, 5)
        sqrt_inv = 1 / np.sqrt(cov.numpy())
        ret_sqrt_inv = self.algorithm._get_chol_inverse(cov)
        np.testing.assert_equal(ret_sqrt_inv.numpy(), sqrt_inv)

    def test_algorithm_works(self):
        self.algorithm.inf_factor = 1.2
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      None, ana_time)
        self.assertFalse(np.any(np.isnan(assimilated_state.values)))

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

    def test_prepare_obs_returns_obs_cov_matrix(self):
        len_time = len(self.obs.time)
        stacked_cov = [self.obs['covariance'].values] * len_time * 2
        block_diag = np.concatenate(stacked_cov)
        returned_cov = self.algorithm._get_obs_cov(
            (self.obs, self.obs)
        )
        np.testing.assert_equal(returned_cov, block_diag)

    def test_prepare_obs_returns_obs_cov_matrix_with_time(self):
        self.obs['covariance'] = self.obs['covariance'].expand_dims(
            time=self.obs['time'], axis=0
        )
        stacked_cov = self.obs['covariance'].stack(
            obs_id=('time', 'obs_grid_1')
        )
        block_diag = np.concatenate([stacked_cov.values, stacked_cov.values])
        returned_cov = self.algorithm._get_obs_cov(
            (self.obs, self.obs)
        )
        np.testing.assert_equal(returned_cov, block_diag)


if __name__ == '__main__':
    unittest.main()
