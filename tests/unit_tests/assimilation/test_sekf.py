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

# Internal modules

from pytassim.assimilation.filter.sekf import SEKFCorr, SEKFUncorr,\
    estimate_inc_uncorr, estimate_inc_corr
from pytassim.testing import dummy_h_jacob, dummy_obs_operator, if_gpu_decorator


logging.basicConfig(level=logging.INFO)


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


class TestSEKFCorr(unittest.TestCase):
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

    def test_get_obs_to_use_comparison_to_all(self):
        obs_grid = np.zeros((10000, 3), dtype=int)
        obs_grid[::2, 0] = 1
        obs_grid[::3, 1] = 1
        obs_grid[::4, 2] = 1

        grid_point = (1, 1, 1)
        right_result = np.all(obs_grid == grid_point, axis=-1)
        np.testing.assert_equal(
            self.algorithm._get_obs_to_use(obs_grid, grid_point), right_result
        )

    def test_get_grid_names_raises_value_error_if_not_multiindex(self):
        state = self.state.unstack('grid')
        state = state.rename({'vgrid': 'grid'})
        with self.assertRaises(ValueError):
            _ = self.algorithm.get_grid_names(state)

    def test_functional(self):
        analysis_mean = self.algorithm.assimilate(
            self.state, (self.obs, ), self.pseudo_state
        ).mean('ensemble')
        analysis_inc = analysis_mean-self.state.mean('ensemble')

        (
            pseudo_obs,
            obs_state,
            obs_cov,
            obs_grid,
            work_state,
            grid_names,
        ) = self.algorithm._prepare_sekf(
            state=self.state, observations=(self.obs, ),
            pseudo_state=self.pseudo_state
        )
        innov = self.algorithm._estimate_departure(pseudo_obs, obs_state)
        for gp in work_state.hgrid.values:
            tmp_innov = innov.sel(obs_grid_1=gp[0])
            obs_to_use = (obs_grid == gp).squeeze()
            tmp_obs_cov = self.algorithm._localize_obs_cov(obs_cov, obs_to_use)
            tmp_pseudo_obs = pseudo_obs.sel(obs_grid_1=gp[0])
            tmp_state = work_state.sel(hgrid=gp)
            tmp_h_jacob = self.algorithm.estimate_h_jacob(
                tmp_state, tmp_pseudo_obs, gp, None
            )
            tmp_b_mat = self.algorithm.estimate_b_matrix(
                tmp_state, tmp_pseudo_obs, gp, None
            )

            tmp_states = self.algorithm._states_to_torch(
                tmp_innov.values, tmp_h_jacob, tmp_b_mat, tmp_obs_cov
            )
            tmp_inc = estimate_inc_corr(*tmp_states).detach().numpy()
            ana_inc = analysis_inc.sel(grid=gp).mean(['var_name', 'time'])
            np.testing.assert_almost_equal(ana_inc, tmp_inc)
    #
    # @if_gpu_decorator
    # def test_functional_gpu(self):
    #     analysis_cpu = self.algorithm.assimilate(
    #         self.state, (self.obs, ), self.pseudo_state
    #     )
    #
    #     self.algorithm.gpu = True
    #     self.algorithm.dtype = torch.float16
    #     analysis_gpu = self.algorithm.assimilate(
    #         self.state, (self.obs, ), self.pseudo_state
    #     )
    #     xr.testing.assert_equal(analysis_cpu, analysis_gpu)

    def test_functional_3d_grid(self):
        new_3d_multiindex = pd.MultiIndex.from_product(
            (np.arange(2), np.arange(5), np.arange(4)),
            names=['lat', 'lon', 'vgrid']
        )
        new_hori_multiindex = pd.MultiIndex.from_product(
            new_3d_multiindex.levels[:-1],
            names=new_3d_multiindex.names[:-1]
        )
        new_hori_multiindex_2 = pd.MultiIndex.from_product(
            new_3d_multiindex.levels[:-1],
            names=['{0:s}_1'.format(n) for n in new_3d_multiindex.names[:-1]]
        )
        state_3d = self.state.copy(deep=True)
        state_3d['grid'] = new_3d_multiindex
        pseudo_state_2d = self.pseudo_state.copy(deep=True)
        pseudo_state_2d['grid'] = new_hori_multiindex
        obs_2d = self.obs.copy(deep=True)
        obs_2d['obs_grid_1'] = new_hori_multiindex
        obs_2d['obs_grid_2'] = new_hori_multiindex_2
        obs_2d.obs.operator = dummy_obs_operator
        analysis_3d = self.algorithm.assimilate(state_3d, (obs_2d, ),
                                                pseudo_state_2d)
        analysis_3d['grid'] = self.state['grid']

        analysis_2d = self.algorithm.assimilate(
            self.state, (self.obs, ), self.pseudo_state
        )
        xr.testing.assert_equal(analysis_3d, analysis_2d)


class TestSEKFUncorr(TestSEKFCorr):
    def setUp(self) -> None:
        super().setUp()
        self.algorithm = SEKFUncorr(
            b_matrix=self.b_matrix, h_jacob=dummy_h_jacob
        )
        self.obs['covariance'] = xr.DataArray(
            np.diag(self.obs.covariance.values),
            coords={
                'obs_grid_1': self.obs.obs_grid_1
            },
            dims=['obs_grid_1']
        )
        self.obs.obs.operator = dummy_obs_operator

    def test_functional(self):
        analysis_mean = self.algorithm.assimilate(
            self.state, (self.obs, ), self.pseudo_state
        ).mean('ensemble')
        analysis_inc = analysis_mean-self.state.mean('ensemble')

        (
            pseudo_obs,
            obs_state,
            obs_cov,
            obs_grid,
            work_state,
            grid_names,
        ) = self.algorithm._prepare_sekf(
            state=self.state, observations=(self.obs, ),
            pseudo_state=self.pseudo_state
        )
        innov = self.algorithm._estimate_departure(pseudo_obs, obs_state)
        for gp in work_state.hgrid.values:
            tmp_innov = innov.sel(obs_grid_1=gp[0])
            obs_to_use = (obs_grid == gp).squeeze()
            tmp_obs_cov = self.algorithm._localize_obs_cov(obs_cov, obs_to_use)
            tmp_pseudo_obs = pseudo_obs.sel(obs_grid_1=gp[0])
            tmp_state = work_state.sel(hgrid=gp)
            tmp_h_jacob = self.algorithm.estimate_h_jacob(
                tmp_state, tmp_pseudo_obs, gp, None
            )
            tmp_b_mat = self.algorithm.estimate_b_matrix(
                tmp_state, tmp_pseudo_obs, gp, None
            )

            tmp_states = self.algorithm._states_to_torch(
                tmp_innov.values, tmp_h_jacob, tmp_b_mat, tmp_obs_cov
            )
            tmp_inc = estimate_inc_uncorr(*tmp_states).detach().numpy()
            ana_inc = analysis_inc.sel(grid=gp).mean(['var_name', 'time'])
            np.testing.assert_almost_equal(ana_inc, tmp_inc)

    # def test_speed_test(self):
    #     dims = 200, 100, 100
    #     multiindex = pd.MultiIndex.from_product(
    #         (np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2])),
    #         names=['lat', 'lon', 'vgrid']
    #     )
    #     rnd = np.random.RandomState(42)
    #     state = xr.DataArray(
    #         rnd.normal(size=(1, 1, 1, np.product(dims))),
    #         dims=self.state.dims,
    #         coords={
    #             'var_name': ['x', ],
    #             'grid': multiindex,
    #             'time': [pd.to_datetime('1992-12-25 08:00 UTC'), ],
    #             'ensemble': [0, ]
    #         }
    #     )
    #     pseudo_state = state.sel(vgrid=0)
    #     obs_perts = rnd.normal(scale=0.1, size=pseudo_state.shape)
    #     obs_values = pseudo_state + obs_perts
    #     obs_values = obs_values.rename({'grid': 'obs_grid_1'}).squeeze(
    #         ['ensemble', 'var_name']
    #     )
    #     obs_stddev = xr.DataArray(
    #         [0.5] * np.product(dims[:-1]),
    #         coords={
    #             'obs_grid_1': obs_values.obs_grid_1
    #         },
    #         dims=['obs_grid_1']
    #     )
    #     obs = xr.Dataset({'observations': obs_values, 'covariance': obs_stddev})
    #     obs.obs.operator = dummy_obs_operator
    #
    #     b_matrix = np.eye(dims[-1]) * 2
    #     h_jacob = np.log(np.arange(dims[-1])[::-1]+1).reshape(1, dims[-1])
    #     self.algorithm.b_matrix = b_matrix
    #     self.algorithm.h_jacob = h_jacob
    #     _ = self.algorithm.assimilate(
    #         state=state, observations=(obs, ), pseudo_state=pseudo_state
    #     )


if __name__ == '__main__':
    unittest.main()
