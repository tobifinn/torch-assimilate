#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2/14/19

Created for torch-assimilate

@author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de

    Copyright (C) {2019}  {Tobias Sebastian Finn}

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
from unittest.mock import MagicMock
import logging
import os
import time

# External modules
import xarray as xr
import torch
import numpy as np
import scipy.spatial.distance

from dask.distributed import LocalCluster, Client

# Internal modules
from pytassim.assimilation.filter.letkf import LETKFCorr, local_etkf
from pytassim.testing import dummy_obs_operator, DummyLocalization
from pytassim.testing.cases import TestDistributedCase
from pytassim.assimilation.filter.letkf_dist import DistributedLETKFCorr, \
    DistributedLETKFUncorr
from pytassim.localization import GaspariCohn
from pytassim.assimilation.filter import etkf_core


logging.basicConfig(level=logging.INFO)
rnd = np.random.RandomState(42)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestLETKFDistributed(TestDistributedCase):
    state_path = os.path.join(DATA_PATH, 'test_state.nc')
    obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')

    def setUp(self):
        self.algorithm = DistributedLETKFCorr(client=self.client)
        self.back_prec = self.algorithm._get_back_prec(len(self.state.ensemble))
        self.localization = DummyLocalization()

    def test_local_etkf_same_results_as_letkf(self):
        letkf_filter = LETKFCorr()
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        state_mean, state_perts = self.state.state.split_mean_perts()
        assimilated_state = letkf_filter.assimilate(self.state, obs_tuple,
                                                    self.state, ana_time)
        delta_ana = assimilated_state - state_mean
        delta_ana = delta_ana.transpose('grid', 'var_name', 'time', 'ensemble')

        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        innov, hx_perts, obs_cov = [
            torch.tensor(s) for s in prepared_states[:-1]
        ]
        state = self.state.transpose('grid', 'var_name', 'time', 'ensemble')
        state_mean, state_perts = state.state.split_mean_perts()
        torch_back_perts = torch.from_numpy(state_perts.values)
        torch_grid = state_mean.grid.values
        for i, _ in enumerate(torch_grid):
            ana_pert, _, _ = local_etkf(
                self.algorithm._gen_weights_func, i, innov, hx_perts, obs_cov,
                self.back_prec, state_mean.grid.values, prepared_states[-1],
                torch_back_perts
            )
            np.testing.assert_almost_equal(
                delta_ana.values[i], ana_pert.numpy()
            )
    #
    # def test_update_state_returns_same_as_letkf(self):
    #     ana_time = self.state.time[-1].values
    #     obs_tuple = (self.obs, self.obs)
    #     assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
    #                                                   self.state, ana_time)
    #
    #     letkf_filter = LETKFCorr()
    #     letkf_state = letkf_filter.assimilate(self.state, obs_tuple, self.state,
    #                                           ana_time)
    #     np.testing.assert_allclose(assimilated_state.values, letkf_state.values)

    def test_localization_works(self):
        localization = DummyLocalization()
        letkf_filter = LETKFCorr(localization=localization)
        self.algorithm.localization = localization
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      self.state, ana_time)
        assimilated_state = assimilated_state.compute()
        letkf_state = letkf_filter.assimilate(self.state, obs_tuple, self.state,
                                              ana_time)
        xr.testing.assert_allclose(assimilated_state, letkf_state)

    def test_letkfuncorr_sets_gen_weights_func(self):
        self.assertEqual(
            DistributedLETKFUncorr(client=self.client)._gen_weights_func,
            etkf_core.gen_weights_uncorr
        )

    def test_letkfuncorr_sets_correlated_to_false(self):
        self.assertFalse(DistributedLETKFUncorr(client=self.client)._correlated)


if __name__ == '__main__':
    unittest.main()
