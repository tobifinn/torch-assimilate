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
from typing import Union, Tuple, Iterable

# External modules
import numpy as np
import torch
import torch.sparse
import scipy.linalg
import xarray as xr
import dask.array as da

# Internal modules
from ..utils import evd, rev_evd


logger = logging.getLogger(__name__)


class ETKFWeightsModule(torch.nn.Module):
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
        return 'ETKFWeightsModule({0})'.format(self.inf_factor)

    def __repr__(self) -> str:
        return 'ETKFWeightsModule'

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
        k_mat = torch.einsum('...ij,...kj->...ik', x, y)
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
        prior_mean = torch.zeros(normed_perts.shape[:-1]+(1,)).to(normed_perts)
        prior_eye = torch.ones(normed_perts.shape[:-1]).to(normed_perts)
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
        w_mean = torch.einsum('...ij,...jk->...ik', cov_analysed, kernel_obs,)

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


class ETKFAnalyser(object):
    """
    Analyser to get analysis perturbations based on given background
    perturbations and normalized observational quantities.
    """
    def __init__(
            self,
            inf_factor: Union[float, torch.Tensor, torch.nn.Parameter] = 1.0
    ):
        self.inf_factor = inf_factor

    def __str__(self) -> str:
        return 'ETKFAnalyser({0})'.format(self.inf_factor)

    def __repr__(self) -> str:
        return 'ETKFAnalyser'

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
        self.gen_weights = ETKFWeightsModule(new_factor)

    @staticmethod
    def _weights_matmul(
            perts: torch.Tensor,
            weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Multiply given ensemble perturbations with ensemble weights.
        """
        ana_perts = torch.einsum('...ig,ij->...jg', perts, weights)
        return ana_perts

    def get_analysis_perts(
            self,
            state_perts: torch.Tensor,
            normed_perts: torch.Tensor,
            normed_obs: torch.Tensor,
            state_grid: np.ndarray,
            obs_grid: np.ndarray
    ) -> torch.Tensor:
        """
        Estimate the analysis perturbations with given data, set inflation
        factor and kernel.
        """
        weights = self.gen_weights(normed_perts, normed_obs)[0]
        weights = weights.detach()
        ana_perts = self._weights_matmul(state_perts, weights)
        return ana_perts

    def __call__(
            self,
            state_perts: torch.Tensor,
            normed_perts: torch.Tensor,
            normed_obs: torch.Tensor,
            state_grid: np.ndarray,
            obs_grid: np.ndarray
    ) -> torch.Tensor:
        return self.get_analysis_perts(state_perts, normed_perts, normed_obs,
                                       state_grid, obs_grid)


class CorrMixin(object):
    _correlated = True

    @staticmethod
    def _get_obs_cov(observations: Iterable[xr.Dataset]) -> np.ndarray:
        """
        Get the observational covariance from given observations.
        """
        cov_stacked_list = []
        for obs in observations:
            len_time = len(obs.time)
            stacked_cov = [obs['covariance'].values] * len_time
            stacked_cov = scipy.linalg.block_diag(*stacked_cov)
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
    def _get_obs_cov(observations: Iterable[xr.Dataset]) -> np.ndarray:
        """
        Get the observational covariance from given observations.
        """
        cov_stacked_list = []
        for obs in observations:
            len_time = len(obs.time)
            stacked_cov = [obs['covariance'].values] * len_time
            stacked_cov = np.concatenate(stacked_cov)
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
