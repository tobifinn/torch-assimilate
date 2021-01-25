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
from copy import deepcopy

# External modules
import xarray as xr
import numpy as np
import pandas as pd

import dask
import dask.array as da

import torch
import torch.jit
import torch.nn
import torch.sparse

import scipy.linalg
import scipy.linalg.blas

# Internal modules
from pytassim.interface.ketkf import KETKF
from pytassim.interface.etkf import ETKF
from pytassim.core.ketkf import KETKFModule
from pytassim import kernels
from pytassim.testing import dummy_obs_operator, if_gpu_decorator



logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestKETKF(unittest.TestCase):
    def setUp(self):
        self.algorithm = KETKF(kernel=kernels.LinearKernel())
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator

    def tearDown(self):
        self.state.close()
        self.obs.close()

    def test_inf_factor_returns_inf_factor_from_core_module(self):
        self.algorithm._core_module.inf_factor = torch.tensor(1.5)
        torch.testing.assert_allclose(self.algorithm.inf_factor,
                                      self.algorithm._core_module.inf_factor)

    def test_inf_factor_sets_inf_factor_from_core_module(self):
        self.algorithm.inf_factor = torch.nn.Parameter(torch.tensor(1.5))
        torch.testing.assert_allclose(
            self.algorithm._core_module.inf_factor,
            torch.nn.Parameter(torch.tensor(1.5))
        )

    def test_inf_factor_sets_old_kernel_to_module(self):
        kernel_id = id(self.algorithm.core_module.kernel)
        self.algorithm.inf_factor = torch.nn.Parameter(torch.tensor(1.5))
        self.assertEqual(id(self.algorithm.core_module.kernel), kernel_id)

    def test_kernel_returns_kernel_from_core_module(self):
        self.algorithm._core_module.kernel = kernels.GaussKernel()
        self.assertEqual(self.algorithm.kernel,
                         self.algorithm._core_module.kernel)

    def test_kernel_sets_kernel_from_core_module(self):
        new_kernel = kernels.GaussKernel()
        self.algorithm.kernel = new_kernel
        self.assertEqual(new_kernel, self.algorithm._core_module.kernel)

    def test_kernel_sets_old_inf_factor_to_module(self):
        inf_factor_id = id(self.algorithm.core_module.inf_factor)
        self.algorithm.kernel = kernels.GaussKernel()
        self.assertEqual(id(self.algorithm.core_module.inf_factor),
                         inf_factor_id)

    def test_inf_factor_sets_ketkf_module(self):
        self.algorithm.inf_factor = torch.nn.Parameter(torch.tensor(1.5))
        self.assertIsInstance(self.algorithm.core_module, KETKFModule)

    def test_kernel_sets_ketkf_module(self):
        self.algorithm.kernel = kernels.GaussKernel()
        self.assertIsInstance(self.algorithm.core_module, KETKFModule)

    def test_ketkf_linear_kernel_same_result_as_etkf(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        ketkf_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                None, ana_time)
        etkf = ETKF(inf_factor=self.algorithm.inf_factor)
        etkf_state = etkf.assimilate(self.state, obs_tuple, None, ana_time)
        np.testing.assert_allclose(ketkf_state.values, etkf_state.values,
                                   rtol=1E-10, atol=1E-10)

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

    def test_ketkf_returns_right_analysis(self):
        self.algorithm.kernel = kernels.GaussKernel()
        sliced_state = self.state.isel(time=[0])
        sliced_obs = self.obs.isel(time=[0])
        sliced_obs.obs.operator = self.obs.obs.operator
        ens_obs = sliced_obs.obs.operator(sliced_obs, sliced_state)
        ens_mean, ens_perts = ens_obs.state.split_mean_perts()
        innovation = sliced_obs['observations']-ens_mean

        norm_innov = sliced_obs.obs.mul_rcinv(innovation)
        norm_innov = torch.from_numpy(norm_innov.values).to(
            self.algorithm.dtype
        ).view(1, -1)

        norm_perts = sliced_obs.obs.mul_rcinv(ens_perts)
        norm_perts = norm_perts.transpose('ensemble', 'time', 'obs_grid_1')
        norm_perts = torch.from_numpy(norm_perts.values).to(
            self.algorithm.dtype
        ).view(10, -1)

        weights = self.algorithm.core_module(norm_perts, norm_innov).numpy()
        weights = xr.DataArray(
            weights,
            coords={
                'ensemble': self.state.indexes['ensemble'],
                'ensemble_new': self.state.indexes['ensemble']
            },
            dims=['ensemble', 'ensemble_new']
        )
        analysis = self.algorithm._apply_weights(sliced_state, weights)
        ret_analysis = self.algorithm.assimilate(state=sliced_state,
                                                 observations=sliced_obs)
        xr.testing.assert_identical(analysis, ret_analysis)

    @if_gpu_decorator
    def test_algorithm_works_gpu(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs.copy())
        self.algorithm.gpu = True
        self.algorithm.kernel = kernels.GaussKernel(torch.tensor(1.0))
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      None, ana_time)
        self.assertFalse(np.any(np.isnan(assimilated_state.values)))

    def test_chunks_return_None(self):
        self.assertIsNone(self.algorithm.chunks)


if __name__ == '__main__':
    unittest.main()
