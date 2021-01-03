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

# External modules
import xarray as xr

import torch

# Internal modules
from pytassim.core.ienks import IEnKSTransformModule, IEnKSBundleModule
from pytassim.core.etkf import ETKFModule
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

    def test_split_weights_splits_prior_correctly(self):
        weights = torch.eye(40)
        splitted_weights = self.module._split_weights(weights, ens_size=40)
        torch.testing.assert_allclose(splitted_weights[0], 0)
        torch.testing.assert_allclose(splitted_weights[1], torch.eye(40))

    def test_split_weights_splits_weights_into_mean_perts(self):
        w_mean = torch.randn(size=(40, 1))
        w_perts = torch.randn(40, 40) * 0.1
        w_perts -= w_perts.mean(dim=1, keepdim=True)
        w_perts += torch.eye(40)
        weights = w_mean + w_perts
        splitted_weights = self.module._split_weights(weights, ens_size=40)
        self.assertEqual(len(splitted_weights), 2)
        torch.testing.assert_allclose(splitted_weights[0], w_mean)
        torch.testing.assert_allclose(splitted_weights[1], w_perts)


if __name__ == '__main__':
    unittest.main()
