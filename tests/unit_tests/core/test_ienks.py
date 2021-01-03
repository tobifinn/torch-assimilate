#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 03.01.21
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import unittest
import logging
import os
from math import sqrt
from typing import Tuple

# External modules
import xarray as xr

import torch

# Internal modules
from pytassim.core.ienks import IEnKSTransformModule, IEnKSBundleModule
from pytassim.core.etkf import ETKFModule
from pytassim.core.utils import svd, rev_svd, evd, rev_evd
from pytassim.testing.dummy import dummy_obs_operator


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        state = xr.open_dataarray(state_path).load().isel(time=0)
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        obs = xr.open_dataset(obs_path).load().isel(time=0)
        obs.obs.operator = dummy_obs_operator
        pseudo_state = obs.obs.operator(obs, state)
        cls.state_perts = state - state.mean('ensemble')
        cls.normed_perts = torch.from_numpy(
            (pseudo_state - pseudo_state.mean('ensemble')).values
        ).float().view(-1, 40)
        cls.normed_obs = torch.from_numpy(
            (obs['observations'] - pseudo_state.mean('ensemble')).values
        ).float().view(1, 40)
        cls.obs_grid = obs['obs_grid_1'].values.reshape(-1, 1)
        cls.state_grid = state.grid.values.reshape(-1, 1)

    def setUp(self) -> None:
        self.module = IEnKSTransformModule(tau=torch.tensor(1.0))

    @staticmethod
    def _construct_weights(
            ens_size: int = 40
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w_mean = torch.randn(size=(ens_size, 1))
        w_perts = torch.randn(ens_size, ens_size) * 0.1
        w_perts -= w_perts.mean(dim=1, keepdim=True)
        w_perts += torch.eye(ens_size)
        weights = w_mean + w_perts
        return weights, w_mean, w_perts

    def test_svd_returns_usv(self):
        _, _, w_perts = self._construct_weights(40)
        u, s, v = torch.svd(w_perts)
        returned_usv = svd(w_perts, reg_value=0.0)
        torch.testing.assert_allclose(returned_usv[0], u)
        torch.testing.assert_allclose(returned_usv[1], s)
        torch.testing.assert_allclose(returned_usv[2], v)

    def test_svd_reg_value_adds_to_diagonal(self):
        _, _, w_perts = self._construct_weights(40)
        w_cov = w_perts @ w_perts.t() / 39.
        u, s, v = svd(w_cov, reg_value=39.)
        svd_way = rev_svd(u, s.pow(-1), v)
        added = (w_cov + 39. * torch.eye(40)).inverse()
        torch.testing.assert_allclose(svd_way, added)

    def test_svd_rev_svd_loops(self):
        _, _, w_perts = self._construct_weights(40)
        u, s, v = svd(w_perts, reg_value=0.0)
        recomposed_perts = rev_svd(u, s, v)
        torch.testing.assert_allclose(recomposed_perts, w_perts)

    def test_split_weights_splits_prior_correctly(self):
        weights = torch.eye(40)
        splitted_weights = self.module._split_weights(weights, ens_size=40)
        torch.testing.assert_allclose(splitted_weights[0], 0)
        torch.testing.assert_allclose(splitted_weights[1], torch.eye(40))

    def test_split_weights_splits_weights_into_mean_perts(self):
        weights, w_mean, w_perts = self._construct_weights(40)
        splitted_weights = self.module._split_weights(weights, ens_size=40)
        self.assertEqual(len(splitted_weights), 2)
        torch.testing.assert_allclose(splitted_weights[0], w_mean)
        torch.testing.assert_allclose(splitted_weights[1], w_perts)

    def test_decomposed_weights_estimates_perts_inverse_and_covariance(self):
        weights, w_mean, w_perts = self._construct_weights(40)
        w_cov = w_perts @ w_perts.t() / 39
        w_perts_inv = w_perts.inverse()
        u, s, v = torch.svd(w_cov)
        w_prec = torch.matmul(u * s.pow(-1), v.t())

        decomposed_weights = self.module._decompose_weights(weights, 40)
        torch.testing.assert_allclose(decomposed_weights[0], w_mean)
        torch.testing.assert_allclose(decomposed_weights[1], w_perts_inv)
        torch.testing.assert_allclose(decomposed_weights[2], w_prec,
                                      rtol=1E-4, atol=1E-4)

    def test_dh_dw_returns_right_matrices(self):
        weights, w_mean, w_perts = self._construct_weights(10)
        w_perts_inv = w_perts.inverse()
        dh_dw = torch.matmul(w_perts_inv, self.normed_perts)
        ret_dh_dw = self.module._get_dh_dw(self.normed_perts,
                                           weights_perts_inv=w_perts_inv)
        torch.testing.assert_allclose(ret_dh_dw, dh_dw)

    def test_get_gradient_returns_gradient_wrt_ensemble_mean(self):
        weights, w_mean, w_perts = self._construct_weights(10)
        w_perts_inv = w_perts.inverse()
        dh_dw = torch.matmul(w_perts_inv, self.normed_perts)
        gradient = 9 * w_mean - torch.matmul(dh_dw, self.normed_obs.t())

        ret_gradient = self.module._get_gradient(
            w_mean, dh_dw, self.normed_obs, ens_size=10
        )
        torch.testing.assert_allclose(ret_gradient, gradient)

    def test_update_covariance_updates_covariance(self):
        self.module.tau = torch.tensor(0.5)
        weights, w_mean, w_perts = self._construct_weights(10)
        w_prec = (w_perts @ w_perts.t() / 9).inverse()
        w_perts_inv = w_perts.inverse()
        dh_dw = torch.matmul(w_perts_inv, self.normed_perts)
        new_prec = dh_dw @ dh_dw.t() + torch.eye(10) * 9
        updated_prec = (1-self.module.tau) * w_prec + self.module.tau * new_prec
        u, s, v = torch.svd(updated_prec)
        updated_cov = torch.matmul(u/s, v.t())
        ret_cov = self.module._update_covariance(w_prec, dh_dw, ens_size=10)[0]
        torch.testing.assert_allclose(ret_cov, updated_cov)

    def test_update_covariance_returns_new_perturbations(self):
        self.module.tau = torch.tensor(0.5)
        weights, w_mean, w_perts = self._construct_weights(10)
        w_prec = (w_perts @ w_perts.t() / 9).inverse()
        w_perts_inv = w_perts.inverse()
        dh_dw = torch.matmul(w_perts_inv, self.normed_perts)
        new_prec = dh_dw @ dh_dw.t() + torch.eye(10) * 9
        updated_prec = (1-self.module.tau) * w_prec + self.module.tau * new_prec
        u, s, v = torch.svd(updated_prec)
        s_perts = (9 / s).sqrt()
        updated_perts = torch.matmul(u * s_perts, v.t())
        returned_perts = self.module._update_covariance(
            w_prec, dh_dw, ens_size=10
        )[1]
        torch.testing.assert_allclose(returned_perts, updated_perts)

    def test_update_covariance_returns_same_perts_as_etkf_if_gauss_newton(self):
        self.module.tau = torch.tensor(1.0)
        kernel_perts = self.normed_perts @ self.normed_perts.t()
        evals, evects, evals_inv = evd(kernel_perts, reg_value=9.0)
        evals_perts = (evals_inv * 9).sqrt()
        weight_perts = rev_evd(evals_perts, evects)
        returned_perts = self.module._update_covariance(
            torch.eye(10), self.normed_perts, ens_size=10
        )[1]
        torch.testing.assert_allclose(returned_perts, weight_perts)

    def test_update_covariance_returns_corresponding_cov_perts(self):
        self.module.tau = torch.tensor(1.0)

        returned_cov, returned_perts = self.module._update_covariance(
            torch.eye(10), self.normed_perts, ens_size=10
        )
        correct_cov = returned_perts @ returned_perts.t() / 9
        torch.testing.assert_allclose(returned_cov, correct_cov)


if __name__ == '__main__':
    unittest.main()
