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
import logging
import os
import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

# External modules
import xarray as xr
import torch
import numpy as np
import scipy.spatial.distance

# Internal modules
from pytassim.assimilation.filter.letkf import LETKFilter, local_etkf
from pytassim.testing import dummy_obs_operator, DummyLocalization
from pytassim.assimilation.filter.letkf_dist import DistributedLETKF
from pytassim.localization import GaspariCohn


logging.basicConfig(level=logging.DEBUG)
rnd = np.random.RandomState(42)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


POOL = ThreadPoolExecutor(max_workers=1)


def dist_func(state_grid, obs_grid):
    state_grid = state_grid[None, None]
    obs_grid = obs_grid[..., None]
    distance = scipy.spatial.distance.cdist(state_grid, obs_grid)
    return distance


class TestLETKFDistributed(unittest.TestCase):
    def setUp(self):
        self.algorithm = DistributedLETKF(POOL)
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        self.back_prec = self.algorithm._get_back_prec(len(self.state.ensemble))
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.obs.obs.operator = dummy_obs_operator

    def test_local_etkf_same_results_as_letkf(self):
        letkf_filter = LETKFilter()
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        state_mean, state_perts = self.state.state.split_mean_perts()
        assimilated_state = letkf_filter.assimilate(self.state, obs_tuple,
                                                    ana_time)
        delta_ana = assimilated_state - state_mean
        delta_ana = delta_ana.transpose('grid', 'var_name', 'time', 'ensemble')

        prepared_states = self.algorithm._prepare(self.state, obs_tuple)
        innov, hx_perts, obs_cov = [
            torch.tensor(s) for s in prepared_states[:-1]
        ]
        state = self.state.transpose('grid', 'var_name', 'time', 'ensemble')
        state_mean, state_perts = state.state.split_mean_perts()
        torch_back_perts = torch.from_numpy(state_perts.values)
        torch_grid = torch.from_numpy(state_mean.grid.values)
        for i, _ in enumerate(torch_grid):
            ana_pert, _, _ = local_etkf(
                i, innov, hx_perts, obs_cov, self.back_prec, None, None,
                torch_back_perts
            )
            np.testing.assert_almost_equal(
                delta_ana.values[i], ana_pert.numpy()
            )

    def test_update_state_returns_same_as_letkf(self):
        letkf_filter = LETKFilter()
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      ana_time)
        letkf_state = letkf_filter.assimilate(self.state, obs_tuple, ana_time)
        xr.testing.assert_allclose(assimilated_state, letkf_state)

    def test_localization_works(self):
        localization = DummyLocalization()
        letkf_filter = LETKFilter(localization=localization)
        self.algorithm.localization = localization
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        assimilated_state = self.algorithm.assimilate(self.state, obs_tuple,
                                                      ana_time)
        letkf_state = letkf_filter.assimilate(self.state, obs_tuple, ana_time)
        xr.testing.assert_allclose(assimilated_state, letkf_state)

    @staticmethod
    def _get_random_obs(nr_points=1000, obs_stddev=1):
        grid_points = np.arange(nr_points)
        observations = rnd.normal(scale=obs_stddev, size=(1, nr_points))
        obs_da = xr.DataArray(
            observations,
            coords={
                'time': [datetime.datetime(1992, 12, 25)],
                'obs_grid_1': grid_points
            },
            dims=['time', 'obs_grid_1']
        )
        obs_cov = xr.DataArray(
            obs_stddev**2 * np.identity(nr_points),
            coords={
                'obs_grid_1': grid_points,
                'obs_grid_2': grid_points
            },
            dims=['obs_grid_1', 'obs_grid_2']
        )
        obs_ds = xr.Dataset({'observations': obs_da, 'covariance': obs_cov})
        return obs_ds

    @staticmethod
    def _get_random_state(nr_points=1000, ensemble_mems=100, state_stddev=2):
        state_data = rnd.normal(
            scale=state_stddev, size=(1, 1, ensemble_mems, nr_points)
        )
        state_da = xr.DataArray(
            state_data,
            coords={
                'var_name': ['x', ],
                'time': [datetime.datetime(1992, 12, 25)],
                'ensemble': np.arange(ensemble_mems),
                'grid': np.arange(nr_points)
            },
            dims=['var_name', 'time', 'ensemble', 'grid']
        )
        return state_da

    def test_speed_test(self):
        localization = GaspariCohn(length_scale=0.001, dist_func=dist_func)
        self.algorithm = LETKFilter(localization)
        state_data = self._get_random_state(nr_points=10000)
        obs_data = self._get_random_obs(nr_points=10000)
        obs_data.obs.operator = dummy_obs_operator

        start_time = time.time()
        _ = self.algorithm.assimilate(state_data, obs_data)
        print('LETKF needs: {0:.1f} s'.format(time.time()-start_time))


if __name__ == '__main__':
    unittest.main()
