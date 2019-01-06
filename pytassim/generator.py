#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 27.09.17
#
# Created for enkf_lorenz
#
#@author: Tobias Sebastian Finn, tobias.sebastian.finn@studium.uni-hamburg.de
#
#    Copyright (C) {2017}  {Tobias Sebastian Finn}
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

# External modules
import numpy as np
import xarray as xr

from tqdm import tqdm

# Internal modules


logger = logging.getLogger(__name__)


def observation_generator(truth, random_gen=np.random.normal, indexes=None,
                          timestep=0.5, time_axis='validtime', *args, **kwargs):
    """
    Generate the observations based on given truth data series. The observations
    are generated directly with given truth series. So no observation operator
    is needed.

    Parameters
    ----------
    truth : xarray.DataArray
        The observations are generated based on this truth DataArray.
    random_gen : numpy random function
        The observation disturbance is generated based on this function. The
        additional arguments and keyword arguments are passed to the function.
    indexes : list or None
        The observations are valid for this given index list. If this is None,
        one observation for every grid point is generated.
    timestep : list(float), float
        Every timestep a observation is generated. The timestep is in the unit
        of the truth time.
    time_axis : str

    Returns
    -------
    observations : xarray.DataArray
        The generated observations.
    """
    logger.info('Started to generate the observations')
    observations = None
    if indexes is None:
        indexes = list(truth['grid'].values)
    elif indexes is int:
        indexes = [indexes, ]
    for k, ind in enumerate(tqdm(indexes)):
        truth_ind = truth.sel(grid=[ind, ], drop=False)
        if isinstance(timestep, (float, int)):
            ts = timestep
        else:
            ts = timestep[k]
        t_start = truth[time_axis].values.min()
        t_end = truth[time_axis].values.max()
        obs_times = np.arange(t_start, t_end+ts, ts)
        logger.debug(obs_times)
        truth_sliced = truth_ind.sel(**{time_axis: obs_times}, method='nearest')
        random_dist = random_gen(size=truth_sliced.shape, *args, **kwargs)
        obs_sliced = truth_sliced + random_dist
        if observations is None:
            observations = obs_sliced
        else:
            observations = xr.concat([observations, obs_sliced], dim='grid')
    observations = observations.rename(grid='obs_grid_1')
    observations = observations.squeeze(['ensemble', 'var_name'])
    return observations
