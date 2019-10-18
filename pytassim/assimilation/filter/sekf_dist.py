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

    def _localize_obs_space(self, pseudo_obs, obs_state, obs_cov, obs_grid,
                            grid_block):
        local_obs_to_use = grid_block.map_blocks(
            lambda grid, gp: self._get_obs_to_use(grid, gp),
            grid=obs_grid, dtype=bool
        )
        local_pseudo_obs = local_obs_to_use.map_blocks(
            lambda obs_idx, state: state[obs_idx],
            state=pseudo_obs, dtype=float
        )
        local_obs_state = local_obs_to_use.map_blocks(
            lambda obs_idx, state: state[obs_idx],
            state=obs_state, dtype=float
        )
        local_obs_cov = local_obs_to_use.map_blocks(
            lambda obs_idx, cov: self._localize_obs_cov(cov, obs_idx),
            dtype=float
        )
        return local_pseudo_obs, local_obs_state, local_obs_cov

    def estimate_h_jacob(self, state, pseudo_obs, grid_block, analysis_time):
        print(state.shape)
        print(state.compute())
        eval_h_jacob = state.map_blocks(
            lambda l_state, block_id: l_state[..., block_id[-1]],
            block_id=True, dtype=int, chunks=tuple(list(state.chunks[:-1])+[1])
        )
        print(eval_h_jacob.compute())
        return eval_h_jacob

    def _get_obs_to_use(self, obs_grid, grid_point):
        obs_grid_equality = obs_grid == grid_point
        obs_to_use = da.ones((obs_grid_equality.shape[0]), dtype=bool)
        for i in range(obs_grid_equality.shape[1]):
            obs_to_use *= obs_grid_equality[..., i]
        return obs_to_use

    def update_state(self, state, observations, pseudo_state, analysis_time):
        (
            pseudo_obs,
            obs_state,
            obs_cov,
            obs_grid,
            work_state,
            grid_names,
        ) = self._prepare_sekf(state, observations, pseudo_state)

        pseudo_obs = self.to_dask_array(pseudo_obs.values)
        obs_state = self.to_dask_array(obs_state)
        obs_cov = self.to_dask_array(obs_cov)
        obs_grid = self.to_dask_array(obs_grid)
        work_state = work_state.chunk(
            {'hgrid': self.chunksize, 'var_name': -1, 'time': -1, 'vgrid': -1}
        )
        grid_iter = self.to_dask_array(work_state.hgrid.values)

        ana_incs = []
        for k, grid_block in enumerate(grid_iter.blocks):
            print(grid_block.compute())
            local_state = work_state.data.blocks[..., k]
            (
                local_pseudo_obs,
                local_obs_state,
                local_obs_cov,
            ) = self._localize_obs_space(
                pseudo_obs, obs_state, obs_cov, obs_grid, grid_block
            )
            local_h_jacob = self.estimate_h_jacob(
                local_state, local_pseudo_obs, grid_block, analysis_time
            )
            local_b_mat = self.estimate_b_matrix(
                local_state, local_pseudo_obs, grid_block, analysis_time
            )
            local_innov = self._estimate_departure(
                tmp_pseudo_obs, tmp_obs_state
            )
            local_states = self._states_to_torch(
                tmp_innov, tmp_h_jacob, tmp_b_mat, tmp_obs_cov
            )
            tmp_inc = self._func_inc(*tmp_states)
            ana_incs.append(tmp_inc.detach().numpy())
        ana_incs = dask.delayed(da.concatenate)(ana_incs, axis=-1)
        analysis = state + ana_incs.compute()
        return analysis
