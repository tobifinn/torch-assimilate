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
import time

# External modules
import xarray as xr
import numpy as np
import torch
import scipy.linalg
import scipy.linalg.blas

# Internal modules
import pytassim.state
import pytassim.observation
from pytassim.assimilation.filter.etkf import ETKFilter
from pytassim.testing import dummy_obs_operator


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


def if_gpu_decorator(func):     # pragma: no cover
    @unittest.skipIf(not torch.cuda.is_available(), "no GPU")
    def newfunc(self, *args, **kwargs):
        func(self, *args, **kwargs)
    return newfunc


class TestETKFilter(unittest.TestCase):
    def setUp(self):
        self.algorithm = ETKFilter()
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator
        self.algorithm._set_back_prec(len(self.state.ensemble))

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
        np.testing.assert_equal(obs_grid, returned_grid)

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

    def test_diagonal_inverse_returns_inverse_of_diagonal_matrix(self):
        obs_tuple = [self.obs, ] * 5
        _, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        _, hx_perts, _ = self.algorithm._prepare_back_obs(self.state, obs_tuple)
        est_c = np.matmul(hx_perts.T, np.linalg.inv(obs_cov))
        self.assertTupleEqual(est_c.shape, hx_perts.T.shape)

        t_obs_cov = torch.tensor(obs_cov)
        t_hx_perts = torch.tensor(hx_perts)
        ret_c = self.algorithm._compute_c_diag(t_hx_perts, t_obs_cov).numpy()
        self.assertTupleEqual(ret_c.shape, hx_perts.T.shape)

    def test_calc_c_chol_calculates_c_based_on_cholesky_decomp(self):
        obs_tuple = [self.obs, ] * 5
        _, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        _, hx_perts, _ = self.algorithm._prepare_back_obs(self.state, obs_tuple)
        alpha = 0
        obs_cov_prod = np.matmul(obs_cov.T, obs_cov)
        obs_hx = np.matmul(obs_cov.T, hx_perts)
        est_c = None
        nr_iters = 0
        while est_c is None:
            try:
                U, lower = scipy.linalg.cho_factor(obs_cov_prod)
                est_c = scipy.linalg.cho_solve((U, lower), obs_hx).T
            except np.linalg.linalg.LinAlgError:
                obs_cov_prod[np.diag_indices_from(obs_cov_prod)] -= alpha
                if alpha == 0:
                    alpha = 0.00001
                else:
                    alpha *= 10
                obs_cov_prod[np.diag_indices_from(obs_cov_prod)] += alpha
            nr_iters += 1
        t_obs_cov = torch.tensor(obs_cov)
        t_hx_perts = torch.tensor(hx_perts)

        ret_c, _ = self.algorithm._compute_c_chol(t_hx_perts, t_obs_cov)
        np.testing.assert_array_almost_equal(est_c, ret_c.numpy())

    def test_calc_inc_alpha_if_singular(self):
        self.obs['covariance'][0, :2] = 0
        self.obs['covariance'][:2, 0] = 0
        obs_tuple = [self.obs, ]
        _, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        _, hx_perts, _ = self.algorithm._prepare_back_obs(self.state, obs_tuple)
        t_obs_cov = torch.tensor(obs_cov)
        t_hx_perts = torch.tensor(hx_perts)

        _, alpha = self.algorithm._compute_c_chol(t_hx_perts, t_obs_cov)
        self.assertGreater(alpha, 0)

    def test_calc_inc_alpha_if_singular_alpha_too_small(self):
        self.obs['covariance'][0, :2] = 0
        self.obs['covariance'][:2, 0] = 0
        obs_tuple = [self.obs, ]
        _, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        _, hx_perts, _ = self.algorithm._prepare_back_obs(self.state, obs_tuple)
        t_obs_cov = torch.tensor(obs_cov)
        t_hx_perts = torch.tensor(hx_perts)

        alpha = np.finfo(np.float64).eps
        _, ret_alpha = self.algorithm._compute_c_chol(t_hx_perts, t_obs_cov,
                                                      alpha)
        self.assertGreater(ret_alpha, alpha)

    def test_calculate_c_returns_same_for_diag(self):
        obs_tuple = [self.obs, ] * 5
        _, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        _, hx_perts, _ = self.algorithm._prepare_back_obs(self.state, obs_tuple)
        obs_cov = torch.tensor(obs_cov)
        hx_perts = torch.tensor(hx_perts)
        diag_c = self.algorithm._compute_c_diag(hx_perts, obs_cov)
        chol_c, _ = self.algorithm._compute_c_chol(hx_perts, obs_cov)
        torch.testing.assert_allclose(diag_c, chol_c)

    def test_calculate_c_calls_diag_if_diag_matrix(self):
        obs_tuple = [self.obs, ] * 5
        _, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        _, hx_perts, _ = self.algorithm._prepare_back_obs(self.state, obs_tuple)
        obs_cov = torch.tensor(obs_cov)
        hx_perts = torch.tensor(hx_perts)
        est_c = self.algorithm._compute_c_diag(hx_perts, obs_cov)
        trg = 'pytassim.assimilation.filter.etkf.ETKFilter._compute_c_diag'
        with patch(trg, return_value=est_c) as c_patch:
            _ = self.algorithm._compute_c(hx_perts, obs_cov)
        c_patch.assert_called_once_with(hx_perts, obs_cov)

    def test_calculate_c_calls_chel_if_else(self):
        self.obs['covariance'][0, :] = 1
        self.obs['covariance'][:, 0] = 1
        obs_tuple = [self.obs, ] * 5
        _, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        _, hx_perts, _ = self.algorithm._prepare_back_obs(self.state, obs_tuple)
        obs_cov = torch.tensor(obs_cov)
        hx_perts = torch.tensor(hx_perts)
        est_c = self.algorithm._compute_c_chol(hx_perts, obs_cov)
        trg = 'pytassim.assimilation.filter.etkf.ETKFilter._compute_c_chol'
        with patch(trg, return_value=est_c) as c_patch:
            _ = self.algorithm._compute_c(hx_perts, obs_cov)
        c_patch.assert_called_once_with(hx_perts, obs_cov)

    def test_calculate_c_multiplies_with_obs_weight(self):
        obs_tuple = [self.obs, ] * 5
        _, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        _, hx_perts, _ = self.algorithm._prepare_back_obs(self.state, obs_tuple)
        obs_len = obs_cov.shape[1]
        obs_weights = np.zeros(obs_len)
        obs_weights[::10] = 1
        obs_cov = torch.tensor(obs_cov)
        hx_perts = torch.tensor(hx_perts)
        obs_weights = torch.tensor(obs_weights)

        est_c = self.algorithm._compute_c_diag(hx_perts, obs_cov) * obs_weights

        ret_c = self.algorithm._compute_c(hx_perts, obs_cov, obs_weights)
        np.testing.assert_array_equal(ret_c.numpy(), est_c.numpy())

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

    def test_calc_precision_divides_by_inflation_factor(self):
        _, obs_cov, _ = self.algorithm._prepare_obs((self.obs, ))
        _, hx_pert, _ = self.algorithm._prepare_back_obs(self.state, (self.obs,))
        nr_obs, ens_size = hx_pert.shape
        obs_cov = torch.tensor(obs_cov)
        hx_pert = torch.tensor(hx_pert)
        estimated_c = self.algorithm._compute_c(
            hx_pert, obs_cov
        )
        prec_obs = torch.mm(estimated_c, hx_pert)
        prec_back = (ens_size-1) * torch.eye(ens_size).double() / 1.1
        precision = prec_back + prec_obs
        self.algorithm.inf_factor = 1.1
        ret_prec = self.algorithm._calc_precision(estimated_c, hx_pert)
        torch.testing.assert_allclose(ret_prec, precision)

    def test_eigendecomposition_returns_eig_eiginv_evects(self):
        obs_state, obs_cov, obs_grid = self.algorithm._prepare_obs((self.obs, ))
        hx_mean, hx_pert, _ = self.algorithm._prepare_back_obs(self.state,
                                                               (self.obs, ))
        hx_pert = torch.tensor(hx_pert).double()
        obs_cov = torch.tensor(obs_cov).double()

        estimated_c = self.algorithm._compute_c(hx_pert, obs_cov)
        prec_ana = self.algorithm._calc_precision(estimated_c, hx_pert)

        evals, evects = np.linalg.eigh(prec_ana.numpy())
        evals[evals < 0] = 0
        evals_inv = 1 / evals
        evects_inv = np.linalg.inv(evects)

        ret_evd = self.algorithm._eigendecomp(prec_ana)
        ret_evals, ret_evects, ret_einv, ret_evects_inv = ret_evd
        np.testing.assert_allclose(ret_evals.numpy(), evals)
        np.testing.assert_allclose(ret_evects.numpy(), evects)
        np.testing.assert_allclose(ret_einv.numpy(), evals_inv)
        np.testing.assert_allclose(ret_evects_inv.numpy(), evects_inv)

    def test_eigendecomposition_can_be_reverted(self):
        obs_state, obs_cov, obs_grid = self.algorithm._prepare_obs((self.obs, ))
        hx_mean, hx_pert, _ = self.algorithm._prepare_back_obs(self.state,
                                                               (self.obs, ))
        hx_pert = torch.tensor(hx_pert).double()
        obs_cov = torch.tensor(obs_cov).double()

        estimated_c = self.algorithm._compute_c(hx_pert, obs_cov)
        prec_ana = self.algorithm._calc_precision(estimated_c, hx_pert)
        ret_evd = self.algorithm._eigendecomp(prec_ana)
        ret_evals, ret_evects, ret_evals_inv, ret_evects_inv = ret_evd
        recon_matrix = torch.matmul(ret_evects, torch.diagflat(ret_evals))
        recon_matrix = torch.matmul(recon_matrix, ret_evects_inv)

        torch.testing.assert_allclose(recon_matrix, prec_ana)

    def test_eigen_cov_variance(self):
        obs_state, obs_cov, obs_grid = self.algorithm._prepare_obs((self.obs, ))
        hx_mean, hx_pert, _ = self.algorithm._prepare_back_obs(self.state,
                                                               (self.obs, ))
        hx_pert = torch.tensor(hx_pert).double()
        obs_cov = torch.tensor(obs_cov).double()

        estimated_c = self.algorithm._compute_c(hx_pert, obs_cov)
        prec_ana = self.algorithm._calc_precision(estimated_c, hx_pert)
        evd = self.algorithm._eigendecomp(prec_ana)
        evals, evects, evals_inv, evects_inv = evd
        cov_ana = torch.matmul(evects, torch.diagflat(evals_inv))
        cov_ana = torch.matmul(cov_ana, evects_inv)

        cov_ana_inv = torch.inverse(prec_ana)
        torch.testing.assert_allclose(cov_ana_inv, cov_ana)

    def test_det_square_root_eigen_returns_weight_perts(self):
        obs_state, obs_cov, obs_grid = self.algorithm._prepare_obs((self.obs, ))
        hx_mean, hx_pert, _ = self.algorithm._prepare_back_obs(self.state,
                                                               (self.obs, ))
        nr_obs, ens_size = hx_pert.shape
        hx_pert = torch.tensor(hx_pert).double()
        obs_cov = torch.tensor(obs_cov).double()
        estimated_c = self.algorithm._compute_c(hx_pert, obs_cov)
        prec_ana = self.algorithm._calc_precision(estimated_c, hx_pert)
        evd = self.algorithm._eigendecomp(prec_ana)
        evals, evects, eval_inv, evects_inv = evd

        w_perts = torch.sqrt((ens_size-1) * eval_inv)
        w_perts = torch.matmul(evects, torch.diagflat(w_perts))
        w_perts = torch.matmul(w_perts, evects_inv)
        ret_perts = self.algorithm._det_square_root_eigen(eval_inv, evects,
                                                          evects_inv)
        torch.testing.assert_allclose(ret_perts, w_perts)
        torch.testing.assert_allclose(torch.sum(ret_perts, dim=0), 1)

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
        evd = self.algorithm._eigendecomp(prec_ana)
        evals, evects, evals_inv, evects_inv = evd
        cov_analysed = torch.matmul(evects, torch.diagflat(evals_inv))
        cov_analysed = torch.matmul(cov_analysed, evects_inv)

        gain = torch.matmul(cov_analysed, estimated_c)

        w_mean = torch.matmul(gain, innov)

        w_perts = self.algorithm._det_square_root_eigen(evals_inv, evects,
                                                        evects_inv)

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
        analysis = analysis.transpose('var_name', 'time', 'ensemble', 'grid')

        ret_analysis = self.algorithm.update_state(self.state, obs_tuple,
                                                   ana_time)

        xr.testing.assert_equal(ret_analysis, analysis)
        self.assertTrue(ret_analysis.state.valid)

    def test_smoothing_doesnt_select_analysis_time(self):
        self.algorithm.smoothing = True
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        prepared_states = [torch.tensor(s) for s in prepared_states]
        w_mean, w_perts = self.algorithm._gen_weights(*prepared_states[:-1])
        state_mean, state_perts = self.state.state.split_mean_perts()
        analysis = self.algorithm._apply_weights(w_mean, w_perts, state_mean,
                                                 state_perts)
        analysis = analysis.transpose('var_name', 'time', 'ensemble', 'grid')

        ret_analysis = self.algorithm.update_state(self.state, obs_tuple,
                                                   ana_time)

        xr.testing.assert_equal(ret_analysis, analysis)

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

    def test_set_back_prec_sets_back_prec_to_tensor(self):
        self.algorithm._back_prec = None
        self.algorithm._set_back_prec(50)

        back_prec = 49 * np.eye(50)
        self.assertIsInstance(self.algorithm._back_prec, torch.Tensor)
        np.testing.assert_array_equal(self.algorithm._back_prec.numpy(),
                                      back_prec)

    @if_gpu_decorator
    def test_set_back_prec_sets_back_prec_on_gpu(self):
        self.algorithm.gpu = True
        self.algorithm._set_back_prec(50)

        back_prec = 49 * np.eye(50)
        self.assertIsInstance(self.algorithm._back_prec, torch.Tensor)
        self.assertTrue(self.algorithm._back_prec.is_cuda)
        np.testing.assert_array_equal(self.algorithm._back_prec.cpu().numpy(),
                                      back_prec)

    def test_prepare_sets_back_prec(self):
        self.algorithm._back_prec = None
        obs_tuple = (self.obs, self.obs.copy())
        ens_size = len(self.state.ensemble)
        back_prec = (ens_size-1) * np.eye(ens_size)

        _ = self.algorithm._prepare(self.state, obs_tuple)
        self.assertIsInstance(self.algorithm._back_prec, torch.Tensor)
        np.testing.assert_array_equal(self.algorithm._back_prec.numpy(),
                                      back_prec)

    def test_update_states_uses_states_to_torch(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        torch_states = self.algorithm._states_to_torch(*prepared_states)
        trg = 'pytassim.assimilation.filter.etkf.ETKFilter._states_to_torch'
        with patch(trg, return_value=torch_states) as torch_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple, ana_time)
        torch_patch.assert_called_once()

    def test_algorithm_works(self):
        self.algorithm.inf_factor = 1.2
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      ana_time)
        self.assertFalse(np.any(np.isnan(assimilated_state.values)))


if __name__ == '__main__':
    unittest.main()
