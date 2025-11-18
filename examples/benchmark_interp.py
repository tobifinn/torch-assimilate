#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 11/17/2025
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2025}  {Tobias Sebastian Finn}
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# System modules
import logging
import datetime
import time
import argparse
from typing import Tuple, Any, Dict

# External modules
import xarray as xr
import numpy as np
import dask
import pandas as pd

# Internal modules
import pytassim
from pytassim.interface import WeightInterpLETKF
from pytassim.localization import GaspariCohn
from pytassim.obs_ops.base_ops import BaseOperator


logger = logging.getLogger(__name__)

rnd = np.random.RandomState(42)

parser = argparse.ArgumentParser(description='LETKF Benchmark')
parser.add_argument(
    '-k', '--ens_size',
    help='The number of ensemble members',
    type=int, default=40
)
parser.add_argument(
    '-l', '--len_grid',
    help='Length of state grid (for each dimension)',
    type=int, default=100
)
parser.add_argument(
    '-n', '--nr_obs',
    help='Number of observations',
    type=int, default=1000
)
parser.add_argument(
    '-r', '--loc_radius',
    help='Localization radius',
    type=int, default=20
)
parser.add_argument(
    '-w', '--num_workers',
    help='Number of parallel workers in dask',
    type=int, default=1
)


def distance_func(x_grid, y_grid):
    # Columns are time, x, y
    dist = np.sqrt(
        (y_grid["x"]- x_grid[1])**2
        + (y_grid["y"] - x_grid[2])**2
    )
    return dist


class NearestOperator(object):
    def __call__(
            self,
            obs_ds: xr.Dataset,
            input_vals: xr.DataArray,
            *args: Tuple[Any],
            **kwargs: Dict[str, Any]
    ) -> xr.DataArray:
        obs_grid_idx = []
        for x, y in obs_ds.indexes["obs_grid_1"]:
            dist_x = input_vals.x.values - x
            dist_y = input_vals.y.values - y
            nearest_idx = np.argmin(dist_x**2+dist_y**2)
            obs_grid_idx.append(nearest_idx)
        pseudo_obs = input_vals.sel(var_name="vel").isel(grid=obs_grid_idx)
        pseudo_obs = pseudo_obs.rename({"grid": "obs_grid_1"})
        pseudo_obs = pseudo_obs.reset_index("obs_grid_1", drop=True)
        pseudo_obs = pseudo_obs.assign_coords(obs_grid_1=obs_ds.obs_grid_1)
        return pseudo_obs       


def get_state_data(len_grid=100, ens_size=50):
    # Create 2D grid
    x_grid = np.arange(len_grid)
    y_grid = np.arange(len_grid)
    grid_1, grid_2 = np.meshgrid(x_grid, y_grid)
    
    # Create MultiIndex
    grid_index = pd.MultiIndex.from_arrays(
        [grid_1.ravel(), grid_2.ravel()], 
        names=['x', 'y']
    )
    
    ens_range = np.arange(ens_size)
    
    data = rnd.normal(size=(1, 1, ens_size, len_grid*len_grid))
    state_array = xr.DataArray(
        data=data,
        coords={
            'var_name': ['vel',],
            'time': [datetime.datetime(1992, 12, 25, 8),],
            'ensemble': ens_range,
            'grid': grid_index
        },
        dims=['var_name', 'time', 'ensemble', 'grid']
    )
    return state_array


def get_truth(len_grid=100):
    # Create 2D grid
    x_grid = np.arange(len_grid)
    y_grid = np.arange(len_grid)
    grid_1, grid_2 = np.meshgrid(x_grid, y_grid)
    
    # Create MultiIndex
    grid_index = pd.MultiIndex.from_arrays(
        [grid_1.ravel(), grid_2.ravel()], 
        names=['x', 'y']
    )

    truth = xr.DataArray(
        rnd.normal(size=(1, len_grid*len_grid)),
        coords={
            'time': [datetime.datetime(1992, 12, 25, 8),],
            'grid': grid_index
        },
        dims=['time', 'grid']
    )
    return truth


def get_obs_data(truth: xr.DataArray, nr_obs=1000):
    # For 2D observations, we'll select random points
    total_points = len(truth.grid)
    indices = rnd.choice(total_points, size=min(nr_obs, total_points), replace=False)
    obs_data = truth.isel(grid=indices)
    obs_data = obs_data.rename({"grid": "obs_grid_1"})

    # Create observation covariance
    obs_cov = xr.ones_like(obs_data)
    
    observations = xr.Dataset(
        {
            'observations': obs_data,
            'covariance': obs_cov
        }
    )
    return observations


def main(
        len_grid=100, nr_obs=1000, ens_size=50, loc_radius=20, num_workers=1
):
    truth = get_truth(len_grid)
    back_state = get_state_data(len_grid, ens_size)
    
    # Create 2D weight grid
    x_weights = back_state.indexes["grid"].get_level_values("x")[::1]
    y_weights = back_state.indexes["grid"].get_level_values("y")[::1]
    weight_grid_index = pd.MultiIndex.from_arrays(
        [x_weights, y_weights], 
        names=['x', 'y']
    )
    
    weight_grid = xr.Dataset(
        coords={
            "time": back_state.indexes["time"],
            "grid": weight_grid_index
        }
    )
    
    obs_state = get_obs_data(truth, nr_obs)
    obs_operator = NearestOperator()
    obs_state.obs.operator = obs_operator

    localization = GaspariCohn(length_scale=loc_radius, dist_func=distance_func)
    letkf = WeightInterpLETKF(
        localization=localization, inf_factor=1.1, weight_grid=None
    )
    start_time = time.time()
    with dask.config.set(num_workers=num_workers): 
        analysis = letkf.assimilate(back_state, obs_state)
        analysis = analysis.compute()
    logger.info(
        'Assimilation duration: {0:.2f} s'.format(time.time() - start_time)
    )
    
    bg_rmse = np.sqrt(((back_state.mean("ensemble")-truth)**2).mean())
    logger.info("Background RMSE: {0:.4f}".format(bg_rmse))
    ana_rmse = np.sqrt(((analysis.mean("ensemble")-truth)**2).mean())
    logger.info("Analysis RMSE: {0:.4f}".format(ana_rmse))


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(
        len_grid=args.len_grid, nr_obs=args.nr_obs, ens_size=args.ens_size,
        loc_radius=args.loc_radius, num_workers=args.num_workers
    )
