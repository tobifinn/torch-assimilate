#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 20.08.20
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
import torch

# Internal modules
from .etkf_core import ETKFWeightsModule, ETKFAnalyser
from ..utils import evd, rev_evd
from pytassim.kernels import LinearKernel


logger = logging.getLogger(__name__)


class KETKFWeightsModule(ETKFWeightsModule):
    def __init__(self, kernel, inf_factor=1.0):
        super().__init__(inf_factor)
        self.add_module('kernel', kernel)

    def _apply_kernel(self, x, y):
        return self.kernel(x, y)

    def forward(self, normed_perts, normed_obs):
        ens_size = normed_perts.shape[0]
        reg_value = (ens_size-1) / self._inf_factor

        k_perts = self._apply_kernel(normed_perts, normed_perts)
        k_part_mean = k_perts.mean(dim=-1, keepdims=True)
        k_part_mean = k_part_mean - k_part_mean.mean(dim=-2, keepdims=True)
        k_perts_centered = k_perts - k_perts.mean(dim=-2, keepdims=True) - \
                           k_part_mean

        evals, evects, evals_inv = evd(k_perts_centered, reg_value)
        cov_analysed = rev_evd(evals_inv, evects)

        k_obs = self._apply_kernel(normed_perts, normed_obs)
        k_obs_centered = k_obs - k_obs.mean(dim=-2, keepdims=True) - k_part_mean
        w_mean = torch.einsum(
            '...ij,...jk->...ik', cov_analysed, k_obs_centered
        )

        square_root_einv = ((ens_size - 1) * evals_inv).sqrt()
        w_perts = rev_evd(square_root_einv, evects)
        weights = w_mean + w_perts
        return weights, w_mean, w_perts, cov_analysed


class KETKFAnalyser(ETKFAnalyser):
    def __init__(self, kernel, inf_factor=1.0):
        self.gen_weights = KETKFWeightsModule(kernel=LinearKernel(),
                                              inf_factor=1.0)
        super().__init__(inf_factor=inf_factor)
        self.kernel = kernel

    @property
    def inf_factor(self):
        return self.gen_weights.inf_factor

    @inf_factor.setter
    def inf_factor(self, new_factor):
        self.gen_weights = KETKFWeightsModule(
            kernel=self.gen_weights.kernel, inf_factor=new_factor
        )

    @property
    def kernel(self):
        return self.gen_weights.kernel

    @kernel.setter
    def kernel(self, new_kernel):
        self.gen_weights = KETKFWeightsModule(
            kernel=new_kernel, inf_factor=self.gen_weights.inf_factor
        )
