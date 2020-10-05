#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 05.10.20
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
import scipy.linalg

# Internal modules
from typing import Iterable

import numpy as np
import torch
import xarray as xr

logger = logging.getLogger(__name__)


class CorrMixin(object):
    _correlated = True

    @staticmethod
    def _get_block_cov_with_time(obs):
        """
        Get stacked covariance from observations with time dimension in
        covariance.
        """
        stacked_cov = obs['covariance'].expand_dims(
            time_2=obs['covariance'].indexes['time']
        )
        stacked_cov = stacked_cov.stack(
            obs_id_1=('time', 'obs_grid_1')
        )
        stacked_cov = stacked_cov.stack(
            obs_id_2=('time_2', 'obs_grid_2')
        )
        return stacked_cov.values

    @staticmethod
    def _get_block_cov_wo_time(obs):
        """
        Get stacked covariance from observations without time dimension in
        covariance.
        """
        len_time = len(obs.time)
        stacked_cov = [obs['covariance'].values] * len_time
        stacked_cov = scipy.linalg.block_diag(*stacked_cov)
        return stacked_cov

    def _get_obs_cov(self, observations: Iterable[xr.Dataset]) -> np.ndarray:
        """
        Get the observational covariance from given observations.
        """
        cov_stacked_list = []
        for obs in observations:
            if 'time' in obs['covariance'].dims:
                stacked_cov = self._get_block_cov_with_time(obs)
            else:
                stacked_cov = self._get_block_cov_wo_time(obs)
            cov_stacked_list.append(stacked_cov)
        obs_cov = scipy.linalg.block_diag(*cov_stacked_list)
        return obs_cov

    @staticmethod
    def _get_chol_inverse(cov: torch.Tensor) -> torch.Tensor:
        """
        Decomposes given covariance with cholesky decomposition and returns the
        inverse of the cholesky decomposition.
        """
        chol_decomp = torch.cholesky(cov)
        chol_inv = chol_decomp.inverse()
        return chol_inv

    @staticmethod
    def _mul_cinv(state: torch.Tensor, cinv: torch.Tensor) -> torch.Tensor:
        """
        Multiplies given tensor with given inverse of the cholesky decomposed
        covariance matrix.
        """
        normed_state = torch.mm(state, cinv)
        return normed_state


class UnCorrMixin(object):
    _correlated = False

    @staticmethod
    def _get_block_cov_wo_time(obs):
        """
        Get stacked covariance from observations without time dimension in
        covariance.
        """
        len_time = len(obs.time)
        stacked_cov = [obs['covariance'].values] * len_time
        stacked_cov = np.concatenate(stacked_cov)
        return stacked_cov

    @staticmethod
    def _get_block_cov_with_time(obs):
        """
        Get stacked covariance from observations with time dimension in
        covariance.
        """
        stacked_cov = obs['covariance'].stack(
            obs_id=('time', 'obs_grid_1')
        )
        stacked_cov = stacked_cov.values
        return stacked_cov

    def _get_obs_cov(self, observations: Iterable[xr.Dataset]) -> np.ndarray:
        """
        Get the observational covariance from given observations.
        """
        cov_stacked_list = []
        for obs in observations:
            if 'time' in obs['covariance'].dims:
                stacked_cov = self._get_block_cov_with_time(obs)
            else:
                stacked_cov = self._get_block_cov_wo_time(obs)
            cov_stacked_list.append(stacked_cov)
        obs_cov = np.concatenate(cov_stacked_list)
        return obs_cov

    @staticmethod
    def _get_chol_inverse(cov: torch.Tensor) -> torch.Tensor:
        """
        Decomposes given covariance with cholesky decomposition and returns the
        inverse of the cholesky decomposition.
        """
        chol_decomp = cov.sqrt()
        chol_inv = 1 / chol_decomp
        return chol_inv

    @staticmethod
    def _mul_cinv(state: torch.Tensor, cinv: torch.Tensor) -> torch.Tensor:
        """
        Multiplies given tensor with given inverse of the cholesky decomposed
        covariance matrix.
        """
        normed_state = state * cinv
        return normed_state
