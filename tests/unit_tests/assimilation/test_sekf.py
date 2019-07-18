#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 7/16/19

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

from dask.distributed import LocalCluster, Client

# Internal modules

from pytassim.assimilation.filter.sekf import SEKFCorr, SEKFUncorr,\
    estimate_inc_uncorr, estimate_inc_corr
from pytassim.testing import dummy_h_jacob, dummy_obs_operator
from pytassim.testing.cases import DistributedCase


logging.basicConfig(level=logging.DEBUG)


BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestKalmanAnalytical(unittest.TestCase):
    def test_analytical_1d_solution(self):
        innov = torch.Tensor([1])
        obs_err = torch.Tensor([np.sqrt(0.5)])
        cov_back = torch.Tensor([[0.5]])
        h_jacob = torch.Tensor([[1]])
        est_inc = estimate_inc_uncorr(innov, h_jacob, cov_back, obs_err)
        np.testing.assert_almost_equal(est_inc.numpy(), np.array([0.5]))

    def test_analytical_2d_solution(self):
        innov = torch.Tensor([1])
        obs_err = torch.Tensor([np.sqrt(0.5)])
        cov_back = torch.Tensor([[0.5, 0.2], [0.2, 0.5]])
        h_jacob = torch.Tensor([[1, 0]])
        est_inc = estimate_inc_uncorr(innov, h_jacob, cov_back, obs_err)
        np.testing.assert_almost_equal(est_inc.numpy(), np.array([0.5, 0.2]))

    def test_analytical_2d_2d_obs_solution(self):
        innov = torch.Tensor([1, 1])
        obs_err = torch.Tensor([np.sqrt(0.5), np.sqrt(0.5)])
        cov_back = torch.Tensor([[0.5, 0.2], [0.2, 0.5]])
        h_jacob = torch.Tensor([[1, 0], [1, 0]])
        est_inc = estimate_inc_uncorr(innov, h_jacob, cov_back, obs_err)
        np.testing.assert_almost_equal(est_inc.numpy(), np.array([2/3, 4/15]))

    def test_analytical_2d_2d_obs_corr_solution(self):
        innov = torch.Tensor([1, 1])
        obs_err = torch.Tensor([np.sqrt(0.5), np.sqrt(0.5)])
        cov_obs = torch.eye(obs_err.size()[0]) * torch.pow(obs_err, 2)
        cov_back = torch.Tensor([[0.5, 0.2], [0.2, 0.5]])
        h_jacob = torch.Tensor([[1, 0], [1, 0]])
        est_inc_uncorr = estimate_inc_uncorr(innov, h_jacob, cov_back, obs_err)
        est_inc_corr = estimate_inc_corr(innov, h_jacob, cov_back, cov_obs)
        np.testing.assert_almost_equal(
            est_inc_corr.numpy(), est_inc_uncorr.numpy()
        )


class TestSEKF(unittest.TestCase):
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
        self.algorithm = SEKFCorr(b_matrix=self.b_matrix, h_jacob=dummy_h_jacob)
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

    def test_get_grid_names(self):
        grid_names = self.state.grid.variable.level_names
        self.assertListEqual(
            grid_names, self.algorithm.get_grid_names(self.state)
        )

    def test_get_grid_names_raises_value_error_if_not_multiindex(self):
        state = self.state.unstack('grid')
        state = state.rename({'vgrid': 'grid'})
        with self.assertRaises(ValueError):
            _ = self.algorithm.get_grid_names(state)

    def test_functional(self):
        analysis = self.algorithm.assimilate(
            self.state, (self.obs, ), self.pseudo_state
        )

        state_det = self.state.mean('ensemble')
        pseudo_state_det = self.pseudo_state.mean('ensemble')
        hori_index, grid_index = self.algorithm.get_horizontal_grid(state_det)
        innov, pseudo_obs, obs_cov, obs_grid = self.algorithm._prepare(
            pseudo_state=pseudo_state_det, observations=(self.obs, )
        )
        for gp in hori_index:
            tmp_innov = innov.sel(obs_grid_1=gp[0])
            obs_to_use = (obs_grid == gp).squeeze()
            tmp_obs_cov = obs_cov[obs_to_use, :][:, obs_to_use]

            tmp_state = state_det.sel(grid=gp)
            tmp_pseudo_obs = pseudo_obs.sel(obs_grid_1=gp[0])
            tmp_h_jacob = self.algorithm.estimate_h_jacob(
                tmp_state, tmp_pseudo_obs
            )
            tmp_b_mat = self.algorithm.estimate_b_matrix(
                tmp_state, tmp_pseudo_obs
            )

            tmp_states = self.algorithm._states_to_torch(
                tmp_innov.values, tmp_h_jacob, tmp_b_mat, tmp_obs_cov
            )
            tmp_inc = estimate_inc_corr(*tmp_states).detach().numpy()
            ana_inc = (analysis.sel(grid=gp)-tmp_state).mean(
                ['var_name', 'time', 'ensemble']
            )
            np.testing.assert_almost_equal(ana_inc, tmp_inc)

    def test_functional_3d_grid(self):
        analysis = self.algorithm.assimilate(
            self.state, (self.obs, ), self.pseudo_state
        )


if __name__ == '__main__':
    unittest.main()
