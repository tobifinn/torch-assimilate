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

# External modules
from xarray import register_dataarray_accessor

# Internal modules


logger = logging.getLogger(__name__)


@register_dataarray_accessor('state')
class ModelState(object):
    """
    This accessor extends an :py:class:`xarray.DataArray` by a ``state``
    property. This state property is used as utility class for an easier
    processing of model states for data assimilation purpose.

    Arguments
    ---------
    xr_da : :py:class:`xarray.DataArray`
        This model state accessor is registered to this array. The array should
        have (``variable``, ``time``, ``ensemble``, ``grid``) as coordinates,
        which are explained below:

            ``variable`` – str
                This coordinate allows to concatenate multiple variables into
                one single array. The coordinate should have :py:class:`str` as
                dtype.

            ``time`` – datetime like
                Different times can be concatenated into this coordinate. This
                coordinate is useful for data assimilation algorithms which
                allow assimilation of time dependent variables. The coordinate
                should be a datetime like dtype.

            ``ensemble`` – int
                This coordinate is used for ensemble based data assimilation
                algorithms. The ensemble members are numbered as
                :py:class:`int`. An integer value of 0 symbolizes a
                deterministic or control run. In some ensemble based algorithms,
                the ensemble coordinate is looped.

            ``grid``
                This coordinate is used to characterize the spatial position
                of one single value within the array. This coordinate can be a
                :py:class:`~pandas.MultiIndex`, where different coordinate
                components are stacked. In some algorithms the analysis is
                calculated for every grid point independently such that the
                algorithm loops over this coordinate.
    """
    def __init__(self, xr_da):
        self.array = xr_da

    @property
    def valid(self):
        pass

    def split_mean_perts(self, dims='ensemble'):
        pass
