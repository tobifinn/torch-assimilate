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
import abc
import numpy as np

# External modules
import torch
import pandas as pd

import dask
import dask.array as da

# Internal modules
from ..dask_mixin import DaskMixin
from .sekf import SEKFCorr


logger = logging.getLogger(__name__)


def localize_states_chunkwise():
    pass


def estimate_inc_chunkwise(
        est_inc_func):
    pass


class DistributedSEKFCorr(DaskMixin, SEKFCorr):
    def __init__(self, b_matrix, h_jacob, client=None, cluster=None,
                 chunksize=10, smoother=True, gpu=False,
                 pre_transform=None, post_transform=None, **kwargs):
        super().__init__(
            b_matrix=b_matrix, h_jacob=h_jacob,
            client=client, cluster=cluster, chunksize=chunksize,
            smoother=smoother, gpu=gpu, pre_transform=pre_transform,
            post_transform=post_transform, **kwargs
        )

    def _localize_states(self, pseudo_obs, obs_state, obs_cov, obs_grid,
                         work_state, grid_block):
        local_state = work_state.sel(hgrid=grid_block)
        local_obs_state = grid_block.map_blocks(
            lambda gp, full_state: full_state[np.all(obs_grid == gp, axis=-1)],
            full_state=obs_state
        )
        print(local_obs_state.compute())
        return local_pseudo_obs, local_obs_state, local_obs_cov, local_state

    def estimate_h_jacob(self, local_states):
        h_jacobs = [
            super().estimate_h_jacob(s, local_states[0][k])
            for k, s in local_states[-1]
        ]
        return h_jacobs

    def update_state(self, state, observations, pseudo_state, analysis_time):
        (
            pseudo_obs,
            obs_state,
            obs_cov,
            obs_grid,
            work_state,
            grid_names,
        ) = self._prepare_sekf(state, observations, pseudo_state)

        obs_state = self.to_dask_array(obs_state)
        obs_cov = self.to_dask_array(obs_cov)
        obs_grid = self.to_dask_array(obs_grid)
        work_state = work_state.chunk(
            {'hgrid': self.chunksize, 'var_name': -1, 'time': -1, 'vgrid': -1}
        )
        grid_iter = self.to_dask_array(work_state.hgrid.values)

        ana_incs = []
        for k, grid_block in enumerate(grid_iter.blocks):
            (
                tmp_pseudo_obs,
                tmp_obs_state,
                tmp_obs_cov,
                tmp_state,
            ) = self._localize_states(pseudo_obs, obs_state, obs_cov, obs_grid,
                                      work_state, grid_block)
            tmp_h_jacob = self.estimate_h_jacob(local_states)
            tmp_b_mat = self.estimate_b_matrix(local_states)
            tmp_innov = self._estimate_departure(
                local_pseudo_obs, local_obs_state
            )
            tmp_states = self._states_to_torch(
                tmp_innov.values, tmp_h_jacob, tmp_b_mat, tmp_obs_cov
            )
            tmp_inc = self._func_inc(*tmp_states)
            ana_incs.append(tmp_inc.detach().numpy())
        ana_incs = dask.delayed(da.concatenate)(ana_incs, axis=-1)
        analysis = state + ana_incs.compute()
        return analysis
