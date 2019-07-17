#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 7/17/19
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
import torch
import pandas as pd

# Internal modules
from .filter import FilterAssimilation
from pytassim.utilities import chol_solve


logger = logging.getLogger(__name__)


def estimate_inc(innov, h_jacob, cov_back, obs_err):
    ht = h_jacob.transpose(-1, -2)
    hb = torch.mm(h_jacob, cov_back)
    innov_prec = torch.mm(hb, ht)
    mat_size = innov_prec.size()[1]
    step = mat_size + 1
    end = mat_size * mat_size
    innov_prec.view(-1)[:end:step] += torch.pow(obs_err, 2)
    norm_innov = chol_solve(innov_prec, innov).t()
    k_dist = torch.mm(cov_back, ht)
    inc_ana = torch.mm(k_dist, norm_innov).squeeze(-1)
    return inc_ana


class SEKF(FilterAssimilation):
    def __init__(self, b_matrix, h_jacob, smoother=True, gpu=False,
                 pre_transform=None, post_transform=None, **kwargs):
        super().__init__(
            smoother=smoother, gpu=gpu, pre_transform=pre_transform,
            post_transform=post_transform
        )
        self._b_matrix = None
        self._h_jacob = None
        self.b_matrix = b_matrix
        self.h_jacob = h_jacob

    def _prepare(self, pseudo_state, observations):
        logger.info('Apply observation operator')
        pseudo_obs, filtered_obs = self._prepare_back_obs(pseudo_state,
                                                          observations)
        logger.info('Concatenate observations')
        obs_state, obs_cov, obs_grid = self._prepare_obs(filtered_obs)
        innov = obs_state - pseudo_obs
        return innov, pseudo_obs, obs_cov, obs_grid

    @staticmethod
    def get_horizontal_grid(state):
        grid_index = state.get_index('grid')
        hori_index = pd.MultiIndex.from_product(
            grid_index.levels[:-1], names=grid_index.names[:-1]
        )
        return hori_index

    def update_state(self, state, observations, pseudo_state, analysis_time):
        innov, pseudo_obs, obs_cov, obs_grid = self._prepare(
            pseudo_state, observations
        )

        grid_hori = self.get_horizontal_grid(state)
        grid_iter = self.to_dask_array(grid_hori.values)

        ana_incs = []
        for k, grid_block in enumerate(grid_iter.blocks):
            sel_innov = innov.sel(grid=grid_block)
            sel_state = state.sel(grid=grid_block)
            state_inc = dask.delayed(estimate_inc_chunkwise)(
                sel_state, sel_innov
            )
            ana_incs.append(state_inc)
        ana_incs = dask.delayed(da.concatenate)(ana_incs, axis=-1)
        analysis = state + ana_incs.compute()
        return analysis