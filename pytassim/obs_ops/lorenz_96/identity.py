#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 10/4/18
#
# Created for neural_assim
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
import numpy as np

import torch

# Internal modules
from pytassim.obs_ops.base_ops import BaseOperator


logger = logging.getLogger(__name__)


class IdentityOperator(BaseOperator):
    def __init__(self, obs_points=None, len_grid=40, random_state=None):
        """
        This linear observation operator is an identity observation operator,
        where observed grid points equal observations.

        Parameters
        ----------
        obs_points : list(int), int or None
            Observed grid points. If this is int, then this number of grid
            points are drawn from grid. If this is a list, observed grid points
            are prescribed. If this is None, all grid points are observed.
            Default is None.
        len_grid : int, optional
            Number of grid points in Lorenz '96 model. Default is 40.
        random_state : :py:class:`numpy.random.RandomState` or None, optional
            This random state can be used for random numbers. Default is None.
        """
        super().__init__(len_grid=len_grid, random_state=random_state)
        self._obs_points = None
        self._sel_obs_points = None
        self.obs_points = obs_points

    @property
    def obs_points(self):
        return self._obs_points

    @obs_points.setter
    def obs_points(self, points):
        if isinstance(points, (int, float)):
            self._sel_obs_points = self.random_state.choice(
                self.len_grid, size=points, replace=False
            )
        elif points is None:
            self._sel_obs_points = np.arange(self.len_grid)
        else:
            self._sel_obs_points = points
        self._obs_points = points

    def obs_op(self, in_array, *args, **kwargs):
        if 'var_name' in in_array.dims:
            in_array = in_array.sel(var_name='x')
        obs_state = in_array.sel(grid=self._sel_obs_points)
        return obs_state

    def torch_operator(self):
        operator = torch.nn.Linear(self.len_grid, len(self._sel_obs_points),
                                   bias=False)
        for param in operator.parameters():
            param.requires_grad = False
        operator.weight.data = torch.zeros_like(operator.weight.data)
        for k, i in enumerate(self._sel_obs_points):
            operator.weight.data[k, i] = 1.
        return operator
