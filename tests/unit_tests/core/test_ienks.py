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
from pytassim.testing.decorators import if_gpu_decorator


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

    def test_update_weights_returns_same_mean_as_etkf_for_prior(self):
        self.module.tau = torch.tensor(1.0)
        etkf = ETKFModule(inf_factor=torch.tensor(1.0))
        etkf_weights_mean = etkf(self.normed_perts, self.normed_obs)[1]
        ienks_mean = self.module._update_weights(
            torch.eye(10), self.normed_perts, self.normed_obs
        )[0]
        torch.testing.assert_allclose(ienks_mean, etkf_weights_mean)

    def test_update_weights_returns_same_perts_as_etkf_for_prior(self):
        self.module.tau = torch.tensor(1.0)
        etkf = ETKFModule(inf_factor=torch.tensor(1.0))
        etkf_perts = etkf(self.normed_perts, self.normed_obs)[2]
        ienks_perts = self.module._update_weights(
            torch.eye(10), self.normed_perts, self.normed_obs
        )[1]
        torch.testing.assert_allclose(ienks_perts, etkf_perts)

    def test_get_gradient_returns_analytical_gradient(self):
        w_mean = torch.tensor([0.1, 0.2, -0.3]).view(3, 1)
        normed_obs = torch.tensor([0.3]).view(1, 1)
        normed_perts = torch.tensor([-0.15, -0.05, 0.2]).view(3, 1)
        gradient = torch.tensor([0.245, 0.415, -0.66]).view(3, 1)
        ret_grad = self.module._get_gradient(
            w_mean, dh_dw=normed_perts, normed_obs=normed_obs, ens_size=3
        )
        torch.testing.assert_allclose(ret_grad, gradient)

    def test_update_weights_returns_right_weights(self):
        self.module.tau = torch.tensor(0.5)
        weights, w_mean, w_perts = self._construct_weights(10)
        dh_dw = torch.matmul(w_perts.inverse(), self.normed_perts)
        grad_obs = -torch.matmul(dh_dw, self.normed_obs.t())
        gradient = 9 * w_mean + grad_obs
        curr_cov = w_perts @ w_perts.t() / 9
        curr_prec = curr_cov.inverse()
        new_prec = dh_dw @ dh_dw.t() + torch.eye(10) * 9
        updated_prec = 0.5 * curr_prec + 0.5 * new_prec
        updated_cov = updated_prec.inverse()
        u, s, v = torch.svd(9 * updated_cov)
        updated_perts = torch.matmul(u * s.sqrt(), v.t())
        w_delta = - 0.5 * torch.matmul(updated_cov, gradient)
        updated_mean = w_mean + w_delta
        ret_mean, ret_perts = self.module._update_weights(
            weights, self.normed_perts, self.normed_obs
        )
        torch.testing.assert_allclose(ret_mean, updated_mean)
        torch.testing.assert_allclose(ret_perts, updated_perts)

    def test_update_weights_returns_given_weights_if_len_0(self):
        weights, w_mean, w_perts = self._construct_weights(10)

        ret_weights = self.module(
            weights, self.normed_perts[..., :0], self.normed_obs[:, :0]
        )
        torch.testing.assert_allclose(ret_weights, weights)

    def test_update_weights_returns_updated_weights(self):
        weights = self._construct_weights(10)[0]
        updated_weights = self.module._update_weights(
            weights, self.normed_perts, self.normed_obs
        )
        updated_weights = updated_weights[0] + updated_weights[1]
        ret_weights = self.module(
            weights, self.normed_perts, self.normed_obs
        )
        torch.testing.assert_allclose(ret_weights, updated_weights)

    def test_update_weights_tests_obs_perts_size(self):
        with self.assertRaises(ValueError):
            self.module(torch.eye(10), self.normed_perts[:, :5],
                        self.normed_obs)

    @if_gpu_decorator
    def test_update_weights_uses_cuda(self):
        device = torch.device('cuda')
        self.module.tau = torch.tensor(0.5)
        prior_weights = torch.eye(10)
        updated_weights = self.module(prior_weights, self.normed_perts,
                                      self.normed_obs)
        self.module.to(device)
        cuda_weights = self.module(
            prior_weights.to(device), self.normed_perts.to(device),
            self.normed_obs.to(device)
        )
        self.assertTrue(cuda_weights.is_cuda)
        torch.testing.assert_allclose(cuda_weights.cpu(), updated_weights)

    def test_module_can_be_compiled(self):
        _ = torch.jit.script(self.module)


if __name__ == '__main__':
    unittest.main()
