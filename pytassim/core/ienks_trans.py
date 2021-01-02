#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 26.12.20
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

# Internal modules
from .base import BaseModule
from .utils import evd, rev_evd, matrix_product


logger = logging.getLogger(__name__)


class IEnKSTransformModule(BaseModule):
    """
    The core module for the transform version of the iterative ensemble
    Kalman smoother (IEnKS).
    This version of the IEnKS uses a learning rate :math:`\\tau` to mitigate
    sampling errors within the linearized observation operator.
    """
    def __init__(
            self,
            tau: Union[torch.Tensor, torch.nn.Parameter] = torch.tensor(1.0)
    ):
        super().__init__()
        self.tau = tau

    def __str__(self) -> str:
        return 'TransformModule(tau={0})'.format(self.tau)

    def __repr__(self) -> str:
        return 'TransformModule'

    @staticmethod
    def _split_weights(
            weights: torch.Tensor,
            ens_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights_deviation = weights - torch.eye(ens_size)
        weights_mean = weights_deviation.mean(dim=1, keepdim=True)
        weights_perts = weights - weights_mean
        return weights_mean, weights_perts

    def _decompose_weights(
            self,
            weights: torch.Tensor,
            ens_size: int,
            ens_mone: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w_mean, w_perts = self._split_weights(weights, ens_size)
        w_cov = torch.mm(w_mean, w_perts.t()) / ens_mone
        evals, evects, evals_inv = evd(w_cov)
        w_perts_inv = rev_evd((ens_mone * evals_inv).sqrt(), evects)
        w_prec = rev_evd(evals_inv, evects)
        return w_mean, w_perts_inv, w_prec

    def _get_dh_dw(
            self,
            normed_perts: torch.Tensor,
            weights_perts_inv: torch.Tensor
    ) -> torch.Tensor:
        dh_dw = matrix_product(normed_perts, weights_perts_inv)
        return dh_dw

    def _get_gradient(
            self,
            w_mean: torch.Tensor,
            dh_dw: torch.Tensor,
            normed_obs: torch.Tensor,
            ens_mone: int
    ) -> torch.Tensor:
        dlobs_dh = -normed_obs
        grad_obs = matrix_product(dlobs_dh, dh_dw)
        grad_back = ens_mone * w_mean
        grad = grad_back + grad_obs
        return grad

    def _update_covariance(
            self,
            w_prec: torch.Tensor,
            dh_dw: torch.Tensor,
            ens_size: int,
            ens_mone: int
    ):
        new_prec = matrix_product(dh_dw, dh_dw)
        new_prec = new_prec + ens_mone * torch.eye(ens_size)
        weights_prec = (1-self.tau) * w_prec + self.tau * new_prec
        evals, evects, evals_inv = evd(weights_prec)
        weights_cov = rev_evd(evals_inv, evects)
        weights_perts = rev_evd((ens_mone * evals_inv).sqrt(), evects)
        return weights_perts, weights_cov

    def _update_weights(
            self,
            weights: torch.Tensor,
            normed_perts: torch.Tensor,
            normed_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ens_size = weights.shape[-2]
        ens_mone = ens_size - 1

        w_mean, w_perts_inv, w_prec = self._decompose_weights(
            weights, ens_size, ens_mone
        )
        dh_dw = self._get_dh_dw(normed_perts, w_perts_inv)
        grad = self._get_gradient(w_mean, dh_dw, normed_obs, ens_mone)
        w_cov, w_perts = self._update_covariance(w_prec, dh_dw, ens_size,
                                                 ens_mone)
        delta_weight = matrix_product(w_cov, grad)
        w_mean = w_mean - self.tau * delta_weight
        return w_mean, w_perts

    def forward(
            self,
            weights: torch.Tensor,
            normed_perts: torch.Tensor,
            normed_obs: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        self._test_sizes(normed_perts, normed_obs)
        if normed_perts.shape[-1] == 0:
            w_mean, w_perts, _ = self._get_prior_weights(
                normed_perts, normed_obs
            )
        else:
            w_mean, w_perts = self._update_weights(
                weights, normed_perts, normed_obs
            )
        weights = w_mean + w_perts
        return weights
