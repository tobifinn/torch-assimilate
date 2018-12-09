#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 26.03.18
#
# Created for torch-assimilate
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
import copy

# External modules
import numpy as np
import xarray as xr
import scipy.linalg

# Internal modules
from ..base import BaseAssimilation


logger = logging.getLogger(__name__)


class FilterAssimilation(BaseAssimilation):
    """
    FilterAssimilation is a base class for all assimilation algorithms, which
    are based on a filtering approach e.g. Ensemble Kalman filters or particle
    filters. This class extends
    :py:class:`~pytassim.assimilation.base.BaseAssimilation` with different
    methods and attributes for a simplified filtering approach.
    """
    @staticmethod
    def _prepare_obs(observations):
        state_stacked_list = []
        cov_stacked_list = []
        for obs in observations:
            stacked_obs = obs['observations'].stack(
                obs_id=('time', 'obs_grid_1')
            )
            len_time = len(obs.time)
            # Cannot use indexing or tiling due to possible rank deficiency
            stacked_cov = [obs['covariance'].values] * len_time
            stacked_cov = scipy.linalg.block_diag(*stacked_cov)
            state_stacked_list.append(stacked_obs)
            cov_stacked_list.append(stacked_cov)
        state_concat = xr.concat(state_stacked_list, dim='obs_id')
        state_values = state_concat.values
        state_grid = state_concat.obs_grid_1.values
        state_covariance = scipy.linalg.block_diag(*cov_stacked_list)
        return state_values, state_covariance, state_grid
