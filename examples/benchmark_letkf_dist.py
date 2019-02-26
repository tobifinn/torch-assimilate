#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 2/26/19
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2019}  {Tobias Sebastian Finn}
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# External modules
import xarray as xr
import numpy as np
from mpi4py.futures import MPIPoolExecutor

# Internal modules
import pytassim
from pytassim.assimilation.filter import DistributedLETKFUncorr
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
    help='Length of state grid', type=int, default=10000
)
parser.add_argument(
    '-n', '--nr_obs',
    help='Number of observations (should be less/equal than state grid length)',
    type=int, default=1000
)
parser.add_argument(
    '-r', '--loc_radius',
    help='Localization radius in grid points',
    type=int, default=20
)
parser.add_argument(
    '-m', '--max_workers',
    help='The number of maximum workers for this MPI call',
    type=int, default=1
)
parser.add_argument(
    '-c', '--chunksize',
    help='The data chunks will have this length',
    type=int, default=100
)
parser.add_argument(
    '-b', '--backend',
    help='Distribution backend (mpi, thread, process)', type=str,
    default='thread'
)


def main(pool, len_grid=10000, nr_obs=1000, ens_size=50, loc_radius=20,
         chunksize=1000):
    back_state = get_state_data(len_grid, ens_size)
    obs_state = get_obs_data(len_grid, nr_obs)
    obs_operator = IdentityOperator(len_grid=len_grid, nr_obs=nr_obs)
    obs_state.obs.operator = obs_operator.get_obs_method

    localization = GaspariCohn(length_scale=loc_radius, dist_func=distance_func)
    letkf = DistributedLETKFUncorr(
        pool=pool, chunksize=chunksize, localization=localization,
        inf_factor=1.1
    )
    start_time = time.time()
    _ = letkf.assimilate(back_state, obs_state)
    logger.info(
        'Assimilation duration: {0:.2f} s'.format(time.time() - start_time)
    )


def get_executor(max_workers=4, backend='thread'):
    if backend == 'mpi':
        executor = MPIPoolExecutor(max_workers=max_workers)
    elif backend == 'process':
        executor = ProcessPoolExecutor(max_workers=max_workers)
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)
    return executor


def distance_func(x_grid, y_grid):
    dist = np.abs(x_grid-y_grid)
    return dist


class IdentityOperator(BaseOperator):
    def __init__(self, len_grid, nr_obs):
        super().__init__(len_grid=len_grid)
        self.nr_obs = nr_obs

    @property
    def obs_grid(self):
        return np.linspace(start=0, stop=self.len_grid, num=self.nr_obs,
                           endpoint=False)

    def obs_op(self, in_array, *args, **kwargs):
        if 'var_name' in in_array.dims:
            in_array = in_array.sel(var_name='x')
        obs_state = in_array.sel(grid=self.obs_grid, method='nearest')
        return obs_state


def get_state_data(len_grid=10000, ens_size=50):
    grid_range = np.arange(len_grid)
    ens_range = np.arange(ens_size)

    data = rnd.normal(size=(1, 1, ens_size, len_grid))
    state_array = xr.DataArray(
        data=data,
        coords={
            'var_name': ['x', ],
            'time': [datetime.datetime(1992, 12, 25, 8), ],
            'ensemble': ens_range,
            'grid': grid_range
        },
        dims=['var_name', 'time', 'ensemble', 'grid']
    )
    return state_array


def get_obs_data(len_grid=10000, nr_obs=1000):
    grid_range = np.linspace(start=0, stop=len_grid, num=nr_obs, endpoint=False)
    data = rnd.normal(size=(1, nr_obs))
    obs_data = xr.DataArray(
        data=data,
        coords={
            'time': [datetime.datetime(1992, 12, 25, 8), ],
            'obs_grid_1': grid_range
        },
        dims=['time', 'obs_grid_1']
    )
    obs_cov = xr.DataArray(
        data=[1, ] * nr_obs,
        coords={
            'obs_grid_1': grid_range
        },
        dims=['obs_grid_1']
    )
    observations = xr.Dataset(
        {
            'observations': obs_data,
            'covariance': obs_cov
        }
    )
    return observations


if __name__ == '__main__':
    args = parser.parse_args()
    with get_executor(
            max_workers=args.max_workers, backend=args.backend
    ) as executor:
        main(pool=executor, len_grid=args.len_grid, nr_obs=args.nr_obs,
             ens_size=args.ens_size, loc_radius=args.loc_radius,
             chunksize=args.chunksize)
