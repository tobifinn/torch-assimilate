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
from unittest.mock import patch

# External modules
import xarray as xr
import numpy as np
import torch
import scipy.linalg

# Internal modules
import pytassim.state
import pytassim.observation
from pytassim.assimilation.filter.etkf import ETKFilter
from pytassim.testing import dummy_obs_operator


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestETKFilter(unittest.TestCase):
    def setUp(self):
        self.algorithm = ETKFilter()
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator

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
        np.testing.assert_equal(obs_grid, returned_grid)

    def test_prepare_obs_returns_obs_cov_matrix(self):
        obs_stacked = self.obs['observations'].stack(
            obs_id=('time', 'obs_grid_1')
        )
        stacked_cov = self.obs['covariance'].sel(
            obs_grid_1=obs_stacked.obs_grid_1.values,
            obs_grid_2=obs_stacked.obs_grid_1.values
        ).values
        block_diag = scipy.linalg.block_diag(stacked_cov, stacked_cov)
        _, returned_cov, _ = self.algorithm._prepare_obs(
            (self.obs, self.obs)
        )
        np.testing.assert_equal(returned_cov, block_diag)

    def test_prepare_obs_works_for_single_obs(self):
        obs_stacked = self.obs['observations'].stack(
            obs_id=('time', 'obs_grid_1')
        )
        stacked_cov = self.obs['covariance'].sel(
            obs_grid_1=obs_stacked.obs_grid_1.values,
            obs_grid_2=obs_stacked.obs_grid_1.values
        )
        returned_obs, returned_cov, returned_grid = self.algorithm._prepare_obs(
            (self.obs, )
        )
        np.testing.assert_equal(returned_obs, obs_stacked.values)
        np.testing.assert_equal(returned_cov, stacked_cov.values)
        np.testing.assert_equal(returned_grid, obs_stacked.obs_grid_1.values)

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
        trg = 'pytassim.assimilation.filter.etkf.ETKFilter._prepare_back_obs'
        with patch(trg, return_value=prepared_state) as prepare_patch:
            _ = self.algorithm._prepare(self.state, obs_tuple)
        prepare_patch.assert_called_once_with(self.state, obs_tuple)

    def test_prepare_calls_prepare_obs_with_filtered_obs(self):
        obs_tuple = (self.obs, self.obs.copy())
        prepared_obs = self.algorithm._prepare_obs((self.obs, ))
        with patch('pytassim.assimilation.filter.etkf.ETKFilter._prepare_obs',
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

    def test_compute_inv_r_obs_solves_linear_system(self):
        _, obs_cov, _ = self.algorithm._prepare_obs((self.obs, ))
        _, hx_pert, _ = self.algorithm._prepare_back_obs(self.state, (self.obs,))
        pinv = np.linalg.pinv(obs_cov)
        c_solved = np.matmul(pinv, hx_pert).T
        obs_cov = torch.tensor(obs_cov)
        hx_pert = torch.tensor(hx_pert)
        c_returned = self.algorithm._compute_c(hx_pert, obs_cov,).numpy()
        np.testing.assert_array_almost_equal(c_returned, c_solved)

    def test_calc_precision_returns_precision(self):
        _, obs_cov, _ = self.algorithm._prepare_obs((self.obs, ))
        _, hx_pert, _ = self.algorithm._prepare_back_obs(self.state, (self.obs,))
        nr_obs, ens_size = hx_pert.shape
        obs_cov = torch.tensor(obs_cov)
        hx_pert = torch.tensor(hx_pert)
        estimated_c = self.algorithm._compute_c(
            hx_pert, obs_cov
        )
        prec_obs = torch.mm(estimated_c, hx_pert)
        prec_back = (ens_size-1) * torch.eye(ens_size).double()
        precision = prec_back + prec_obs
        ret_prec = self.algorithm._calc_precision(estimated_c, hx_pert)
        torch.testing.assert_allclose(ret_prec, precision)

    def test_det_square_root_returns_weight_perts(self):
        obs_state, obs_cov, obs_grid = self.algorithm._prepare_obs((self.obs, ))
        hx_mean, hx_pert, _ = self.algorithm._prepare_back_obs(self.state,
                                                               (self.obs, ))
        nr_obs, ens_size = hx_pert.shape
        hx_pert = torch.tensor(hx_pert).double()
        obs_cov = torch.tensor(obs_cov).double()
        estimated_c = self.algorithm._compute_c(hx_pert, obs_cov)
        prec_ana = self.algorithm._calc_precision(estimated_c, hx_pert)
        evals, evects = torch.eig(prec_ana, eigenvectors=True)
        evals = evals[:, 0]
        evals_inv = 1 / evals
        w_perts = torch.sqrt((ens_size-1) * evals_inv)
        w_perts = torch.matmul(evects.t(), torch.diagflat(w_perts))
        w_perts = torch.matmul(w_perts, evects)

        ret_perts = self.algorithm._det_square_root(evals_inv, evects)
        torch.testing.assert_allclose(ret_perts, w_perts)

    def test_eigendecomposition_returns_eig_eiginv_evects(self):
        obs_state, obs_cov, obs_grid = self.algorithm._prepare_obs((self.obs, ))
        hx_mean, hx_pert, _ = self.algorithm._prepare_back_obs(self.state,
                                                               (self.obs, ))
        hx_pert = torch.tensor(hx_pert).double()
        obs_cov = torch.tensor(obs_cov).double()

        estimated_c = self.algorithm._compute_c(hx_pert, obs_cov)
        prec_ana = self.algorithm._calc_precision(estimated_c, hx_pert)

        evals, evects = torch.eig(prec_ana, eigenvectors=True)
        evals = evals[:, 0]
        evals_inv = 1 / evals

        ret_evals, ret_evects, ret_einv = self.algorithm._eigendecomp(
            prec_ana)
        torch.testing.assert_allclose(ret_evals, evals)
        torch.testing.assert_allclose(ret_evects, evects)
        torch.testing.assert_allclose(ret_einv, evals_inv)

    def test_gen_weights_returns_mean_pert_weight(self):
        obs_state, obs_cov, obs_grid = self.algorithm._prepare_obs((self.obs, ))
        hx_mean, hx_pert, _ = self.algorithm._prepare_back_obs(self.state,
                                                               (self.obs, ))
        hx_mean = torch.tensor(hx_mean).double()
        hx_pert = torch.tensor(hx_pert).double()
        obs_state = torch.tensor(obs_state).double()
        obs_cov = torch.tensor(obs_cov).double()
        innov = obs_state - hx_mean

        estimated_c = self.algorithm._compute_c(hx_pert, obs_cov)
        prec_ana = self.algorithm._calc_precision(estimated_c, hx_pert)

        evals, evects, evals_inv = self.algorithm._eigendecomp(prec_ana)

        cov_analysed = torch.matmul(evects.t(), torch.diagflat(evals_inv))
        cov_analysed = torch.matmul(cov_analysed, evects)

        gain = torch.matmul(cov_analysed, estimated_c)

        w_mean = torch.matmul(gain, innov)

        w_perts = self.algorithm._det_square_root(evals_inv, evects)

        ret_mean, ret_perts = self.algorithm._gen_weights(
            innov, hx_pert, obs_cov
        )
        torch.testing.assert_allclose(ret_mean, w_mean)
        torch.testing.assert_allclose(ret_perts, w_perts)

    def test_update_state_uses_prepare_function(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        with patch('pytassim.assimilation.filter.etkf.ETKFilter._prepare',
                   return_value=prepared_states) as prepare_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple,
                                            self.state.time[-1].values)
        prepare_patch.assert_called_once_with(self.state, obs_tuple)

    def test_update_state_calls_gen_weights_once(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        prepared_states = [torch.tensor(s) for s in prepared_states]
        weights = self.algorithm._gen_weights(*prepared_states[:-1])
        with patch('pytassim.assimilation.filter.etkf.ETKFilter._gen_weights',
                   return_value=weights) as weights_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple,
                                            self.state.time[-1].values)
        weights_patch.assert_called_once()

    def test_weights_matmul_applies_matmul(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        prepared_states = [torch.tensor(s) for s in prepared_states]
        w_mean, w_perts = self.algorithm._gen_weights(*prepared_states[:-1])
        weights = w_mean + w_perts
        state_mean, state_perts = self.state.state.split_mean_perts()
        ana_perts = xr.apply_ufunc(
            np.matmul, state_perts, weights.numpy(),
            input_core_dims=[['ensemble'], []], output_core_dims=[['ensemble']],
            dask='parallelized'
        )
        ana_perts = ana_perts.transpose('var_name', 'time', 'ensemble', 'grid')

        ret_perts = self.algorithm._weights_matmul(state_perts, weights.numpy())
        xr.testing.assert_equal(ret_perts, ana_perts)

    def test_apply_weights_applies_weights_to_state(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        prepared_states = [torch.tensor(s) for s in prepared_states]
        w_mean, w_perts = self.algorithm._gen_weights(*prepared_states[:-1])
        weights = w_mean + w_perts
        state_mean, state_perts = self.state.state.split_mean_perts()
        ana_perts = self.algorithm._weights_matmul(state_perts, weights.numpy())
        analysis = state_mean + ana_perts

        ret_analysis = self.algorithm._apply_weights(w_mean, w_perts,
                                                     state_mean, state_perts)

        xr.testing.assert_equal(ret_analysis, analysis)

    def test_update_state_applies_weights(self):
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        prepared_states = [torch.tensor(s) for s in prepared_states]
        w_mean, w_perts = self.algorithm._gen_weights(*prepared_states[:-1])
        state_mean, state_perts = self.state.state.split_mean_perts()
        analysis = self.algorithm._apply_weights(w_mean, w_perts, state_mean,
                                                 state_perts)
        with patch('pytassim.assimilation.filter.etkf.ETKFilter._apply_weights',
                   return_value=analysis) as apply_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple,
                                            self.state.time[-1].values)
        apply_patch.assert_called_once()

    def test_update_state_returns_analysis(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        prepared_states = [torch.tensor(s) for s in prepared_states]
        w_mean, w_perts = self.algorithm._gen_weights(*prepared_states[:-1])
        back_state = self.state.sel(time=[ana_time, ])
        state_mean, state_perts = back_state.state.split_mean_perts()
        analysis = self.algorithm._apply_weights(w_mean, w_perts, state_mean,
                                                 state_perts)

        ret_analysis = self.algorithm.update_state(self.state, obs_tuple,
                                                   ana_time)

        xr.testing.assert_equal(ret_analysis, analysis)


if __name__ == '__main__':
    unittest.main()
