#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 14.03.18
#
# Created for tf-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2018}  {Tobias Sebastian Finn}
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
import abc

# External modules
from xarray import register_dataset_accessor

# Internal modules


logger = logging.getLogger(__name__)


@register_dataset_accessor('obs')
class Observation(object):
    """
    This class represents an observation subset. Within an observation subset,
    the observations can be spatial correlated. Different observation subsets
    are per definition uncorrelated. This class is registered under
    :py:attr:`xarray.Dataset.obs`, where the here defined attributes and methods
    can be accessed. The assimilation algorithms will use the abstract method
    :py:meth:`xarray.Dataset.obs.operator` to convert the given model field into
    an observation equivalent. Thus, the `operator` method needs to be
    overwritten for every observation subset independently.

    Arguments
    ---------
    xr_ds : :py:class:`~xarray.Dataset`
        This :py:class:`~xarray.Dataset` is used for the observation operator.
        The dataset needs two variables:

            observations
                (time, obs_grid_1), the actual observations

            covariance
                (obs_grid_1, obs_grid_2), the covariance between different
                observations

        `obs_grid_1` and `obs_grid_2` are the same, but due to internals of
        xarray they are saved under different coordinates. It is possible to
        define different observation times within the `time` coordinate of the
        given :py:class:`~xarray.Dataset`.
    """
    def __init__(self, xr_ds):
        self.ds = xr_ds

    def _check_dims(self):
        necessary_dims = (
            'time', 'obs_grid_1', 'obs_grid_2'
        )
        keys_avail = all(
            True if d in tuple(self.ds.dims.keys()) else False
            for d in necessary_dims
        )
        return keys_avail

    @property
    def valid(self):
        keys_avail = self._check_dims()
        if keys_avail:
            return True
        else:
            return False

    @abc.abstractmethod
    def operator(self, state):
        pass
