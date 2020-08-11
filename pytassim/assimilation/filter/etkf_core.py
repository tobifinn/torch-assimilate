#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 2/14/19
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
import numpy as np
import torch

# Internal modules
from ..utils import evd, rev_evd


logger = logging.getLogger(__name__)


class ETKFWeightsModule(torch.nn.Module):
    def __init__(self, inf_factor=1.0):
        super().__init__()
        self._inf_factor = None
        self.inf_factor = inf_factor

    @property
    def inf_factor(self):
        return self._inf_factor

    @inf_factor.setter
    def inf_factor(self, new_factor):
        if isinstance(new_factor, (torch.Tensor, torch.nn.Parameter)):
            self._inf_factor = new_factor
        else:
            self._inf_factor = torch.tensor(new_factor)

    @staticmethod
    def _dot_product(x, y):
        k_mat = torch.mm(x, y.t())
        return k_mat

    def forward(self, normed_perts, normed_obs):
        ens_size = normed_perts.shape[0]
        reg_value = (ens_size-1) / self._inf_factor
        kernel_perts = torch.mm(normed_perts, normed_perts.t())
        evals, evects, evals_inv, evects_inv = evd(kernel_perts, reg_value)
        cov_analysed = rev_evd(evals_inv, evects, evects_inv)

        kernel_obs = torch.mm(normed_perts, normed_obs.t())
        w_mean = torch.mm(cov_analysed, kernel_obs).squeeze()

        square_root_einv = ((ens_size - 1) * evals_inv).sqrt()
        w_perts = rev_evd(square_root_einv, evects, evects_inv)
        weights = w_mean + w_perts
        return weights, w_mean, w_perts, cov_analysed


class _CorrMixin(object):
    _correlated = True

    @staticmethod
    def _get_obs_cov(observations):
        cov_stacked_list = []
        for obs in observations:
            len_time = len(obs.time)
            stacked_cov = [obs['covariance'].values] * len_time
            stacked_cov = scipy.linalg.block_diag(*stacked_cov)
            cov_stacked_list.append(stacked_cov)
        obs_cov = scipy.linalg.block_diag(*cov_stacked_list)
        return obs_cov

    @staticmethod
    def _get_chol_inverse(cov):
        chol_decomp = torch.cholesky(cov)
        chol_inv = chol_decomp.inverse()
        return chol_inv

    @staticmethod
    def _normalise_cinv(state, cinv):
        normed_state = state @ cinv
        return normed_state


class _UnCorrMixin(object):
    _correlated = False

    @staticmethod
    def _get_obs_cov(observations):
        cov_stacked_list = []
        for obs in observations:
            len_time = len(obs.time)
            stacked_cov = [obs['covariance'].values] * len_time
            stacked_cov = np.concatenate(stacked_cov)
            cov_stacked_list.append(stacked_cov)
        obs_cov = np.concatenate(cov_stacked_list)
        return obs_cov

    @staticmethod
    def _normalise_cinv(state, cinv):
        normed_state = state * cinv
        return normed_state

    @staticmethod
    def _get_chol_inverse(cov):
        sqrt_cov = cov.sqrt()
        sqrt_inv = 1 / sqrt_cov
        return sqrt_inv