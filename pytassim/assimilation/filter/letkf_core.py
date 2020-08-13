#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 13.08.20
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}
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
import torch.jit

import xarray as xr

# Internal modules
from .etkf_core import ETKFAnalyser, ETKFWeightsModule


logger = logging.getLogger(__name__)


class LETKFAnalyser(ETKFAnalyser):
    def __init__(self, localization=None, inf_factor=1.0):
        self._gen_weights = None
        super().__init__(inf_factor)
        self.localization = localization

    @property
    def gen_weights(self):
        return self._gen_weights

    @gen_weights.setter
    def gen_weights(self, new_module):
        if new_module is None:
            self._gen_weights = None
        elif isinstance(new_module, ETKFWeightsModule):
            self._gen_weights = torch.jit.script(new_module)
        else:
            raise TypeError('Given weights module is not a valid '
                            '`ETKFWeightsModule or None!')

    def _localise_obs(self, grid_point, normed_perts, normed_obs, obs_grid):
        if self.localization is None:
            return normed_perts, normed_obs
        else:
            use_obs, obs_weights = self.localization.localize_obs(
                grid_point, obs_grid
            )
            obs_weights = torch.as_tensor(
                obs_weights[use_obs], dtype=normed_perts.dtype
            ).sqrt()
            normed_perts = normed_perts[..., use_obs] * obs_weights
            normed_obs = normed_obs[..., use_obs] * obs_weights
            return normed_perts, normed_obs

    def get_analysis_perts(self, state_perts, normed_perts, normed_obs,
                           obs_grid):
        grid_first = state_perts.transpose('grid', ...)
        analysis_perts = []
        for sub_perts in grid_first:
            loc_perts, loc_obs = self._localise_obs(
                sub_perts.grid.values, normed_perts, normed_obs, obs_grid
            )
            loc_analysis_perts = super().get_analysis_perts(
                sub_perts, loc_perts, loc_obs
            )
            analysis_perts.append(loc_analysis_perts)
        analysis_perts = xr.concat(analysis_perts, dim='grid')
        analysis_perts = analysis_perts.transpose(*state_perts.dims)
        return analysis_perts
