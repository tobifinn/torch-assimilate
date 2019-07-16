#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 7/16/19
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

# External modules
import pandas as pd

# Internal modules
from .base import BaseAssimilation
from .dask_mixin import DaskMixin


logger = logging.getLogger(__name__)


class BaseSEKF(DaskMixin, BaseAssimilation):
    def __init__(self, b_matrix, h_jacob, client=None, cluster=None,
                 chunksize=10, smoother=True, gpu=False,
                 pre_transform=None, post_transform=None, **kwargs):
        super().__init__(
            client=client, cluster=cluster, chunksize=chunksize,
            smoother=smoother, gpu=gpu, pre_transform=pre_transform,
            post_transform=post_transform
        )
        self._b_matrix = None
        self._h_jacob = None
        self.b_matrix = b_matrix
        self.h_jacob = h_jacob

    @staticmethod
    def get_horizontal_grid(state):
        grid_index = state.get_index('grid')
        hori_index = pd.MultiIndex.from_product(
            grid_index.levels[:-1], names=grid_index.names[:-1]
        )
        return hori_index
