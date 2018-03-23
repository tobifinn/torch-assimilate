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
    This accessor extends an `xarray.DataArray` by a `state` property. This
    state property is used as utility class for an easier processing of model
    states for data assimilation purpose.
    """
    def __init__(self, xr_da):
        self.array = xr_da

    @property
    def valid(self):
        pass

    def split_mean_perts(self, dims='ensemble'):
        pass
