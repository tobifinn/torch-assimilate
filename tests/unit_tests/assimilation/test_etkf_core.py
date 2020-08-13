#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 13.08.20

Created for torch-assimilate

@author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de

    Copyright (C) {2020}  {Tobias Sebastian Finn}

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
import torch
import numpy as np
import xarray as xr

# Internal modules
from pytassim.assimilation.filter.etkf_core import ETKFWeightsModule, \
    ETKFAnalyser
from pytassim.assimilation.utils import evd, rev_evd


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


def _create_matrices():
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


class TestETKFModule(unittest.TestCase):
    def setUp(self):
        self.module = ETKFWeightsModule()
        self.state, self.obs = _create_matrices()
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
        out_dot = self.module._dot_product(self.normed_perts, self.normed_perts)
        torch.testing.assert_allclose(out_dot, right_dot_product)

    def test_differentiable(self):
        normed_perts = torch.nn.Parameter(self.normed_perts.clone())
        self.assertIsNone(normed_perts.grad)
        ret_val = self.module(normed_perts, self.normed_obs)[0]
        ret_val.mean().backward()
        self.assertIsNotNone(normed_perts.grad)
        self.assertIsInstance(normed_perts.grad, torch.Tensor)

    def test_right_cov(self):
        ret_kernel = self.module._dot_product(self.normed_perts,
                                              self.normed_perts)
        ret_evd = evd(ret_kernel, 1)
        evals, evects, evals_inv, evects_inv = ret_evd

        cov_analysed = torch.matmul(evects, torch.diagflat(evals_inv))
        cov_analysed = torch.matmul(cov_analysed, evects_inv)

        right_cov = np.array([
            [0.75, 0.25],
            [0.25, 0.75]
        ])

        np.testing.assert_array_almost_equal(cov_analysed, right_cov)

    def test_rev_evd(self):
        ret_kernel = self.module._dot_product(self.normed_perts,
                                              self.normed_perts)
        evals, evects, evals_inv, evects_inv = evd(ret_kernel, 1)
        right_rev = torch.mm(evects, torch.diagflat(evals))
        right_rev = torch.mm(right_rev, evects_inv)

        ret_rev = rev_evd(evals, evects, evects_inv)
        torch.testing.assert_allclose(ret_rev, right_rev)

    def test_right_w_eigendecomposition(self):
        ret_prec = self.normed_perts @ self.normed_perts.t()
        evals, evects = np.linalg.eigh(ret_prec)
        evals = evals + 1
        evals_inv_sqrt = np.diagflat(np.sqrt(1/evals))
        w_pert = np.dot(evals_inv_sqrt, evects.T)
        w_pert = np.dot(evects, w_pert)

        ret_perts = self.module(self.normed_perts, self.normed_obs)[2]
        np.testing.assert_array_almost_equal(ret_perts.numpy(), w_pert)

    def test_returns_w_mean(self):
        correct_gain = np.array([0.5, -0.5])
        correct_wa = correct_gain * 0.2
        ret_wa = self.module(self.normed_perts, self.normed_obs)[1]
        np.testing.assert_array_almost_equal(ret_wa.numpy(), correct_wa)

    def test_returns_w_perts(self):
        right_cov = np.array([
            [0.75, 0.25],
            [0.25, 0.75]
        ])
        return_perts = self.module(self.normed_perts, self.normed_obs)[2]
        return_perts = return_perts.numpy()
        ret_cov = np.matmul(return_perts, return_perts.T)
        np.testing.assert_array_almost_equal(ret_cov, right_cov)

    def test_returns_w_cov(self):
        right_cov = np.array([
            [0.75, 0.25],
            [0.25, 0.75]
        ])
        return_cov = self.module(self.normed_perts, self.normed_obs)[3]
        return_cov = return_cov.numpy()
        np.testing.assert_array_almost_equal(return_cov, right_cov)

    def test_returns_weights(self):
        weights, w_mean, w_perts, _ = self.module(self.normed_perts,
                                                  self.normed_obs)
        torch.testing.assert_allclose(weights, w_mean+w_perts)

    def test_weights_ens_mean(self):
        weights, w_mean, _, _ = self.module(self.normed_perts, self.normed_obs)
        eval_mean = (weights-torch.eye(self.normed_perts.shape[0])).mean(dim=0)
        torch.testing.assert_allclose(eval_mean, w_mean)


class TestETKFAnalyser(unittest.TestCase):
    def setUp(self) -> None:
        state, obs = _create_matrices()
        innov = (obs['observations']-state.mean('ensemble'))
        innov = innov.values.reshape(-1)
        hx_perts = state.values.reshape(2, 1)
        obs_cov = obs['covariance'].values
        prepared_states = [innov, hx_perts, obs_cov]
        torch_states = [torch.from_numpy(s).float() for s in prepared_states]
        innov, hx_perts, obs_cov = torch_states
        self.obs_cinv = torch.cholesky(obs_cov).inverse()
        self.normed_perts = hx_perts
        self.normed_obs = innov
        self.analyser = ETKFAnalyser(1.0)

    def test_normalise_cinv_multiplies_cinv(self):
        state = torch.zeros(10, 5).normal_()
        perts = torch.zeros(100, 5).normal_()
        cov = (perts.t() @ perts) / 99
        chol_decomp = np.linalg.cholesky(cov.numpy())
        cinv = torch.from_numpy(np.linalg.inv(chol_decomp)).float()
        norm_state = torch.mm(state, cinv)
        ret_state = self.analyser._normalise_cinv(state, cinv)
        torch.testing.assert_allclose(ret_state, norm_state)



if __name__ == '__main__':
    unittest.main()
