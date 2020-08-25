#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 20.08.20

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
import xarray as xr
import torch
import torch.jit
import numpy as np
import dask.array as da

# Internal modules
from pytassim.assimilation.filter.ketkf_core import KETKFWeightsModule, \
    KETKFAnalyser
from pytassim.assimilation.utils import evd, rev_evd
from pytassim import kernels
from pytassim.testing import dummy_obs_operator


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestKETKFWeightsModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        state = xr.open_dataarray(state_path).load().isel(time=0)
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        obs = xr.open_dataset(obs_path).load().isel(time=0)
        obs.obs.operator = dummy_obs_operator
        pseudo_state = obs.obs.operator(state)
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
        self.module = KETKFWeightsModule(kernel=kernels.LinearKernel(),
                                         inf_factor=1.0)

    def test_kernel_is_added_as_module(self):
        new_kernel = kernels.LinearKernel()
        self.module = KETKFWeightsModule(kernel=new_kernel)
        named_modules = dict(self.module.named_modules())
        self.assertIn('kernel', named_modules.keys())
        self.assertEqual(named_modules['kernel'], new_kernel)

    def test_apply_kernel_uses_kernel(self):
        new_kernel = kernels.RBFKernel(gamma=10)
        self.module = KETKFWeightsModule(kernel=new_kernel)
        ret_k = self.module._apply_kernel(self.normed_perts, self.normed_obs)
        right_k = new_kernel(self.normed_perts, self.normed_obs)
        torch.testing.assert_allclose(ret_k, right_k)

    def test_fordward_returns_weights(self):
        new_kernel = kernels.RBFKernel(gamma=10)
        self.module = KETKFWeightsModule(kernel=new_kernel)

        k_perts = new_kernel(self.normed_perts, self.normed_perts)
        m_k_perts = k_perts.mean(dim=-2, keepdims=True)
        k_perts_m = k_perts.mean(dim=-1, keepdims=True)
        m_k_perts_m = k_perts_m.mean(dim=-1, keepdims=True)
        k_perts_centered = k_perts - k_perts_m - m_k_perts + m_k_perts_m
        reg_value = 9.
        evals, evects, evals_inv = evd(k_perts_centered, reg_value)
        cov_analysed = rev_evd(evals_inv, evects)
        k_obs = new_kernel(self.normed_perts, self.normed_obs)
        m_k_obs = k_obs.mean(dim=-2, keepdims=True)
        k_obs_centered = k_obs - k_perts_m - m_k_obs + m_k_perts_m
        w_mean = torch.mm(cov_analysed, k_obs_centered)
        sqrt_evals = (9. * evals_inv).sqrt()
        w_perts = rev_evd(sqrt_evals, evects)
        weights = w_mean + w_perts

        ret_weight_stats = self.module(self.normed_perts, self.normed_obs)

        torch.testing.assert_allclose(ret_weight_stats[0], weights)
        torch.testing.assert_allclose(ret_weight_stats[1], w_mean)
        torch.testing.assert_allclose(ret_weight_stats[2], w_perts)
        torch.testing.assert_allclose(ret_weight_stats[3], cov_analysed)


if __name__ == '__main__':
    unittest.main()
