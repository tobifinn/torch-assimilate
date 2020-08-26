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

import torch
import torch.jit
import torch.nn
import torch.sparse

import scipy.linalg
import scipy.linalg.blas

# Internal modules
import pytassim.state
import pytassim.observation
from pytassim.assimilation.filter.etkf import ETKFCorr
from pytassim.assimilation.filter.ketkf import KETKFCorr
from pytassim import kernels
from pytassim.testing import dummy_obs_operator, if_gpu_decorator



logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestKETKF(unittest.TestCase):
    def setUp(self):
        self.algorithm = KETKFCorr(kernel=kernels.LinearKernel())
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

    def test_kernel_gets_kernel_from_analyser(self):
        old_id = deepcopy(id(self.algorithm.kernel))
        new_kernel = kernels.RBFKernel(gamma=10.)
        self.algorithm._analyser.kernel = new_kernel
        self.assertEqual(id(new_kernel), id(self.algorithm.kernel))
        self.assertNotEqual(old_id, id(self.algorithm.kernel))

    def test_linear_gives_same_result_as_etkf(self):
        etkf = ETKFCorr()
        obs_tuple = (self.obs, self.obs)
        etkf_analysis = etkf.assimilate(self.state, obs_tuple)
        ketkf_analysis = self.algorithm.assimilate(self.state, obs_tuple)
        xr.testing.assert_allclose(ketkf_analysis, etkf_analysis)


if __name__ == '__main__':
    unittest.main()
