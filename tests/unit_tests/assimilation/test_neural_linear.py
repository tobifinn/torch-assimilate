#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12/11/18

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

# External modules#
import xarray as xr

# Internal modules
from pytassim.assimilation.neural import NeuralAssimilation
from pytassim.assimilation.neural.models.linear import DeepAssimilation


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestNeuralLinear(unittest.TestCase):
    def setUp(self):
        self.model = DeepAssimilation()
        self.algorithm = NeuralAssimilation(model=self.model, smoother=False)
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load().isel(var_name=[0, ])
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).sel(obs_grid_1=range(5),
                                                 obs_grid_2=range(5))

    def test_model_assimilate_can_be_used(self):
        analysis = self.algorithm.assimilate(self.state, self.obs)


if __name__ == '__main__':
    unittest.main()
