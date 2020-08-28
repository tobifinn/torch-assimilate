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
from typing import Type, Union, Iterable, Tuple

# External modules
import torch

# Internal modules
from .etkf_core import ETKFWeightsModule, ETKFAnalyser
from ..utils import evd, rev_evd
from pytassim.kernels import LinearKernel
from pytassim.kernels.base_kernels import BaseKernel


logger = logging.getLogger(__name__)


class KETKFWeightsModule(ETKFWeightsModule):
    """
    This weights module estimate ensemble weights based on the kernelized
    ensemble transform Kalman filter and represents the inner core of the KETKF.

    Parameters
    ----------
    kernel : child of :py:class:`pytassim.kernels.base_kernels.BaseKernel`
        This kernel is used to estimate the ensemble weights.
    inf_factor : torch.Tensor or torch.nn.Parameter, optional
        The prior covariance is inflated with this inflation factor. This
        inflation factor is also a type of l2-regularization for the Gaussian
        processes and specifies the uncertainty of the prior ensemble weights.
    """
    def __init__(
            self,
            kernel: Type[BaseKernel],
            inf_factor: Union[float, torch.Tensor, torch.nn.Parameter]
                      = torch.tensor(1.0)
    ):
        super().__init__(inf_factor)
        self.add_module('kernel', kernel)

    def __str__(self):
        return 'KETKFWeightsModule({0:s}, {1})'.format(str(self.kernel),
                                                       self.inf_factor)

    def __repr__(self):
        return 'KETKFWeights({0:s})'.format(repr(self.kernel))

    def _apply_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Applies set kernel on given `x` and `y`.
        """
        return self.kernel(x, y)

    def _estimate_weights(
            self,
            normed_perts: torch.Tensor,
            normed_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate the ensemble weights with set kernel and inflation factor, and
        given perturbations and observations.
        """
        ens_size = normed_perts.shape[0]
        reg_value = (ens_size-1) / self._inf_factor

        k_perts = self._apply_kernel(normed_perts, normed_perts)
        k_part_mean = torch.mean(k_perts, dim=-1, keepdim=True)
        k_part_mean = k_part_mean - torch.mean(k_part_mean, dim=-2,
                                               keepdim=True)
        k_perts_centered = k_perts - torch.mean(k_perts, dim=-2,
                                                keepdim=True) - k_part_mean

        evals, evects, evals_inv = evd(k_perts_centered, reg_value)
        cov_analysed = rev_evd(evals_inv, evects)

        k_obs = self._apply_kernel(normed_perts, normed_obs)
        k_obs_centered = k_obs - torch.mean(k_obs, dim=-2, keepdim=True) - \
                         k_part_mean
        w_mean = torch.einsum(
            '...ij,...jk->...ik', cov_analysed, k_obs_centered
        )

        square_root_einv = ((ens_size - 1) * evals_inv).sqrt()
        w_perts = rev_evd(square_root_einv, evects)
        return w_mean, w_perts, cov_analysed


class KETKFAnalyser(ETKFAnalyser):
    """
    Analyser to get analysis perturbations based on set kernel, and given
    background perturbations and normalized observational quantities.
    """
    def __init__(
            self,
            kernel: Type[BaseKernel],
            inf_factor: Union[float, torch.Tensor, torch.nn.Parameter]
                      = torch.tensor(1.0)
    ):
        self.gen_weights = KETKFWeightsModule(kernel=kernel,
                                              inf_factor=inf_factor)
        super().__init__(inf_factor=inf_factor)
        self.kernel = kernel

    def __str__(self) -> str:
        return 'KETKFAnalyser({0:s}, {1})'.format(str(self.kernel),
                                                  self.inf_factor)

    def __repr__(self) -> str:
        return 'KETKFAnalyser({0:s})'.format(repr(self.kernel))

    @property
    def inf_factor(self) -> Union[float, torch.Tensor, torch.nn.Parameter]:
        return self.gen_weights.inf_factor

    @inf_factor.setter
    def inf_factor(
            self,
            new_factor: Union[float, torch.Tensor, torch.nn.Parameter]
    ):
        """
        Sets a new inflation factor.
        """
        self.gen_weights = KETKFWeightsModule(
            kernel=self.gen_weights.kernel, inf_factor=new_factor
        )

    @property
    def kernel(self) -> Type[BaseKernel]:
        return self.gen_weights.kernel

    @kernel.setter
    def kernel(self, new_kernel: Type[BaseKernel]):
        """
        Sets a new kernel.
        """
        self.gen_weights = KETKFWeightsModule(
            kernel=new_kernel, inf_factor=self.gen_weights.inf_factor
        )
