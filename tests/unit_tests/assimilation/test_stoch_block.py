#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12/18/18

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

# External modules
import xarray as xr
import torch

# Internal modules
from pytassim.assimilation.neural.models.stoch_block import StochBlock


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestStochasticBlock(unittest.TestCase):
    def setUp(self):
        self.block = StochBlock(40, (8, 40))
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()

    def test_recon_loss_returns_mse_loss(self):
        test_state = torch.from_numpy(self.state.values.reshape(-1, 40))
        target_state = test_state - 1
        right_mse = torch.mean((test_state - target_state) ** 2)
        returned_mse = self.block.recon_loss(test_state, target_state)
        torch.testing.assert_allclose(returned_mse, right_mse)


if __name__ == '__main__':
    unittest.main()
