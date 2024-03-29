#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 29.12.20
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Tuple

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


class BaseModule(torch.nn.Module):
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

    @staticmethod
    def _view_as_2d(tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) < 2:
            tensor = tensor.view(1, -1)
        else:
            tensor = tensor.view(-1, tensor.shape[-1])
        return tensor

    @staticmethod
    def _get_prior_weights(
            ens_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get prior weights. The perturbations and covariance matrix are
        already inflated by set inflation factor.
        """
        ens_size = ens_tensor.shape[-2]
        prior_mean = torch.zeros(ens_size, 1).to(ens_tensor)
        prior_eye = torch.ones(ens_size).to(ens_tensor)
        prior_eye = torch.diag_embed(prior_eye)
        prior_cov = prior_eye / (ens_size-1)
        prior_perts = prior_eye
        return prior_mean, prior_perts, prior_cov
