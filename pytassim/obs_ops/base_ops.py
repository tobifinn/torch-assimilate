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
import abc
from typing import Union, Any, Tuple, Dict

# External modules
import xarray as xr
import numpy as np
import torch

# Internal modules


logger = logging.getLogger(__name__)


class BaseOperator(object):
    def __init__(
            self,
            len_grid: int = 40,
            random_state: Union[None, np.random.RandomState] = None
    ):
        """
        This is a BaseClass for observation operators. These observation
        operators are used to map a model state to observations.

        Parameters
        ----------
        len_grid : int, optional
            Number of grid points in Lorenz '96 model. Default is 40.
        random_state : :py:class:`numpy.random.RandomState` or None, optional
            This random state can be used for random numbers. Default is None.
        """
        self.len_grid = len_grid
        self.random_state = random_state

    def __call__(
            self,
            obs_ds: xr.Dataset,
            input_vals: xr.DataArray,
            *args: Tuple[Any],
            **kwargs: Dict[str, Any]
    ) -> xr.DataArray:
        pseudo_obs = self.obs_op(input_vals, *args, **kwargs)
        pseudo_obs = pseudo_obs.rename(grid='obs_grid_1')
        pseudo_obs['time'] = obs_ds.time.values
        pseudo_obs['obs_grid_1'] = obs_ds.obs_grid_1.values
        return pseudo_obs

    @abc.abstractmethod
    def obs_op(
            self,
            in_array: xr.DataArray,
            *args: Tuple[Any],
            **kwargs: Dict[str, Any]
    ) -> xr.DataArray:
        pass

    @abc.abstractmethod
    def torch_operator(self) -> torch.Tensor:
        pass
