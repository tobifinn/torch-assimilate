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
from .utils import evd, rev_evd


logger = logging.getLogger(__name__)


class ETKFModule(torch.nn.Module):
    """
    Module to create ETKF weights based on PyTorch.
    This module estimates weight statistics with given perturbations and
    observations.
    """
    def __init__(
            self,
            inf_factor: Union[float, torch.Tensor, torch.nn.Parameter] = 1.0
    ):
        super().__init__()
        self._inf_factor = None
        self.inf_factor = inf_factor

    def __str__(self) -> str:
        return 'ETKFCore({0})'.format(self.inf_factor)

    def __repr__(self) -> str:
        return 'ETKFCore'

    @property
    def inf_factor(self) -> Union[float, torch.Tensor, torch.nn.Parameter]:
        return self._inf_factor

    @inf_factor.setter
    def inf_factor(
            self, new_factor: Union[float, torch.Tensor, torch.nn.Parameter]
    ):
        """
        Sets a new inflation factor.
        """
        if isinstance(new_factor, (torch.Tensor, torch.nn.Parameter)):
            self._inf_factor = new_factor
        else:
            self._inf_factor = torch.tensor(new_factor)

    @staticmethod
    def _test_sizes(normed_perts: torch.Tensor, normed_obs: torch.Tensor):
        """
        Tests if sizes between perturbations and observations match.
        """
        if normed_perts.shape[-1] != normed_obs.shape[-1]:
            raise ValueError(
                'Observational size between ensemble ({0:d}) and observations '
                '({1:d}) do not match!'.format(
                    normed_perts.shape[-1], normed_obs.shape[-1]
                )
            )
        if normed_perts.shape[:-2] != normed_obs.shape[:-2]:
            raise ValueError(
                'Batch sizes between ensemble {0} and observations {1} do not '
                'match!'.format(
                    tuple(normed_perts.shape[:-2]), tuple(normed_obs.shape[:-2])
                )
            )

    @staticmethod
    def _apply_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply set kernel matrix, here the dot product, to given tensors.
        """
        k_mat = torch.mm(x, y.t())
        return k_mat

    def _get_prior_weights(
            self,
            normed_perts: torch.Tensor,
            normed_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get prior weights. The perturbations and covariance matrix are
        already inflated by set inflation factor.
        """
        ens_size = normed_perts.shape[-2]
        prior_mean = torch.zeros(ens_size, 1).to(normed_perts)
        prior_eye = torch.ones(ens_size).to(normed_perts)
        prior_eye = torch.diag_embed(prior_eye)
        prior_cov = self._inf_factor / (ens_size-1) * prior_eye
        prior_perts = self._inf_factor.sqrt() * prior_eye
        return prior_mean, prior_perts, prior_cov

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
        reg_value = (ens_size-1) / self._inf_factor
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
        else:
            w_mean, w_perts, cov_analysed = self._estimate_weights(
                normed_perts, normed_obs
            )
        weights = w_mean + w_perts
        return weights, w_mean, w_perts, cov_analysed
