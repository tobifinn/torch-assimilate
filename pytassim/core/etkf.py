#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 08.12.20
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union, Tuple

# External modules
import torch
import torch.nn

# Internal modules
from .base import BaseModule
from .utils import evd, rev_evd, matrix_product


logger = logging.getLogger(__name__)


class ETKFModule(BaseModule):
    """
    Module to create ETKF weights based on PyTorch.
    This module estimates weight statistics with given perturbations and
    observations.
    """
    def __init__(
            self,
            inf_factor: Union[torch.Tensor, torch.nn.Parameter] =
            torch.tensor(1.0)
    ):
        super().__init__()
        self.inf_factor = inf_factor

    def __str__(self) -> str:
        return 'ETKFCore({0})'.format(self.inf_factor)

    def __repr__(self) -> str:
        return 'ETKFCore'

    @staticmethod
    def _apply_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply set kernel matrix, here the matrix product, to given tensors.
        """
        k_mat = matrix_product(x, y)
        return k_mat

    def _estimate_weights(
            self,
            normed_perts: torch.Tensor,
            normed_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimates the weights with set inflation factor, _apply_kernel method
        and given data.
        """
        ens_size = normed_perts.shape[-2]
        reg_value = (ens_size-1) / self.inf_factor
        kernel_perts = self._apply_kernel(normed_perts, normed_perts)
        evals, evects, evals_inv = evd(kernel_perts, reg_value)
        cov_analysed = rev_evd(evals_inv, evects)

        kernel_obs = self._apply_kernel(normed_perts, normed_obs)
        w_mean = torch.mm(cov_analysed, kernel_obs)

        square_root_einv = ((ens_size - 1) * evals_inv).sqrt()
        w_perts = rev_evd(square_root_einv, evects)
        return w_mean, w_perts, cov_analysed

    def forward(
            self,
            normed_perts: torch.Tensor,
            normed_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the ensemble weights for given inflation factor, _apply_kernel
        method and data.
        If the perturbations and observations are empty, the inflated prior
        weights are returned.
        """
        self._test_sizes(normed_perts, normed_obs)
        if normed_perts.shape[-1] == 0:
            w_mean, w_perts, cov_analysed = self._get_prior_weights(
                normed_perts, normed_obs
            )
            w_perts = w_perts * self.inf_factor.sqrt()
            cov_analysed = cov_analysed * self.inf_factor
        else:
            w_mean, w_perts, cov_analysed = self._estimate_weights(
                normed_perts, normed_obs
            )
        weights = w_mean + w_perts
        return weights, w_mean, w_perts, cov_analysed
