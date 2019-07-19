#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 7/19/19

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
import logging
import os

# External modules
import xarray as xr
import numpy as np
import pandas as pd
import torch

# Internal modules
from pytassim.assimilation.filter.sekf import SEKFCorr
from pytassim.assimilation.filter.sekf_dist import DistributedSEKFCorr
from pytassim.testing import dummy_h_jacob, dummy_obs_operator, if_gpu_decorator
from pytassim.testing.cases import DistributedCase


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestSEKFDistributed(DistributedCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        cls.state = xr.open_dataarray(state_path).load()
        cls.obs = xr.open_dataset(obs_path).load()
        cls.obs.obs.operator = dummy_obs_operator

    def setUp(self) -> None:
        self.b_matrix = np.identity(4) * 0.5
        self.algorithm = DistributedSEKFCorr(
            client=self.client, b_matrix=self.b_matrix, h_jacob=dummy_h_jacob
        )
        self.state = self.verticalize_state(self.state, 4)
        pseudo_state = self.state.sel(vgrid=0).copy()
        self.pseudo_state = pseudo_state.rename({'lon': 'grid'})
        self.obs = self.obs.sel(obs_grid_1=(self.obs.obs_grid_1 % 4 == 0),
                                obs_grid_2=(self.obs.obs_grid_2 % 4 == 0))
        self.obs['obs_grid_1'] = self.obs.obs_grid_1 // 4
        self.obs['obs_grid_2'] = self.obs.obs_grid_2 // 4
        self.obs.obs.operator = dummy_obs_operator

    @ staticmethod
    def verticalize_state(state, vert_dim=4) -> xr.DataArray:
        state = state.copy()
        vert_grid = state['grid'].values % vert_dim
        hori_grid = state['grid'].values // vert_dim
        zipped_grid = [t for t in zip(hori_grid, vert_grid)]
        multi_grid = pd.MultiIndex.from_tuples(
            zipped_grid, names=['lon', 'vgrid']
        )
        state['grid'] = multi_grid
        return state

    def test_distributed_gives_same_solution_as_undistributed(self):
        analysis = self.algorithm.assimilate(
            self.state, (self.obs, ), self.pseudo_state
        )

        undistributed = SEKFCorr(b_matrix=self.b_matrix,
                                 h_jacob=self.algorithm.h_jacob)
        analysis_undist = undistributed.assimilate(
            self.state, (self.obs, ), self.pseudo_state
        )
        xr.testing.assert_equal(analysis, analysis_undist)


if __name__ == '__main__':
    unittest.main()
