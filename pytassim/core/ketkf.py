#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 15.12.20
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Type, Union, Iterable, Tuple

# External modules
import torch

# Internal modules
from .etkf import ETKFModule
from .utils import evd, rev_evd
from pytassim.kernels.base_kernels import BaseKernel


logger = logging.getLogger(__name__)


class KETKFModule(ETKFModule):
    """
    This KETKF module estimate ensemble weights based on the kernelized
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
            kernel: BaseKernel,
            inf_factor: Union[float, torch.Tensor, torch.nn.Parameter]
                      = torch.tensor(1.0)
    ):
        super().__init__(inf_factor)
        self.add_module('kernel', kernel)

    def __str__(self):
        return 'KETKFModule({0:s}, {1})'.format(str(self.kernel),
                                                self.inf_factor)

    def __repr__(self):
        return 'KETKF({0:s})'.format(repr(self.kernel))

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
        reg_value = (ens_size-1) / self.inf_factor

        k_perts = self._apply_kernel(normed_perts, normed_perts)
        k_partial_mean = torch.mean(k_perts, dim=-1, keepdim=True)
        k_partial_mean = k_partial_mean - torch.mean(k_partial_mean, dim=-2,
                                                     keepdim=True)
        k_perts_centered = k_perts - torch.mean(k_perts, dim=-2,
                                                keepdim=True) - k_partial_mean

        evals, evects, evals_inv = evd(k_perts_centered, reg_value)
        cov_analysed = rev_evd(evals_inv, evects)

        k_obs = self._apply_kernel(normed_perts, normed_obs)
        k_obs_centered = k_obs - torch.mean(k_obs, dim=-2, keepdim=True)
        k_obs_centered = k_obs_centered - k_partial_mean
        w_mean = torch.mm(cov_analysed, k_obs_centered)

        square_root_einv = ((ens_size - 1) * evals_inv).sqrt()
        w_perts = rev_evd(square_root_einv, evects)
        return w_mean, w_perts, cov_analysed
