#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 03.01.21
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union, Iterable, Callable, List

# External modules
import torch
import xarray as xr

# Internal modules
from .variational import VarAssimilation
from pytassim.core.ienks import IEnKSTransformModule, IEnKSBundleModule
from pytassim.transform import BaseTransformer
from pytassim.utilities.decorators import ensure_tensor, bound_tensor


logger = logging.getLogger(__name__)


class IEnKSTransform(VarAssimilation):
    def __init__(
            self,
            model: Callable,
            tau: int = 1.0,
            max_iter: int = 10,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None,
    ):
        super().__init__(
            model=model,
            max_iter=max_iter,
            smoother=smoother,
            gpu=gpu,
            pre_transform=pre_transform,
            post_transform=post_transform
        )
        self.tau = tau

    def __str__(self):
        return 'IEnKSTransform(tau={0})'.format(str(self.tau.item()))

    def __repr__(self):
        return 'IEnKSTransform({0})'.format(repr(self.tau.item()))

    @property
    def tau(self):
        return self._core_module.tau

    @tau.setter
    @ensure_tensor
    @bound_tensor(min_val=0.0, max_val=1.0)
    def tau(self, new_tau):
        self._core_module = IEnKSTransformModule(tau=new_tau)

    def get_model_weights(self, weights: xr.DataArray) -> xr.DataArray:
        return weights

    def estimate_weights(
            self,
            state: xr.DataArray,
            weights: xr.DataArray,
            filtered_obs: List[xr.Dataset],
            ens_obs: List[xr.DataArray],
    ) -> xr.DataArray:
        innovations, ens_obs_perts = self._get_obs_space_variables(
            ens_obs, filtered_obs
        )
        weights = xr.apply_ufunc(
            self.module,
            weights,
            ens_obs_perts,
            innovations,
            input_core_dims=[
                ['ensemble', 'ensemble_new'],
                ['ensemble', 'obs_id'],
                ['obs_id']
            ],
            dask='parallelized',
            output_core_dims=[['ensemble', 'ensemble_new']],
            output_dtypes=[float],
        )
        return weights


class IEnKSBundle(IEnKSTransform):
    def __init__(
            self,
            model: Callable,
            tau: int = 1.0,
            epsilon: int = 1E-4,
            max_iter: int = 10,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None,
    ):
        self._core_module = IEnKSBundleModule()
        super().__init__(
            model=model,
            tau=tau,
            max_iter=max_iter,
            smoother=smoother,
            gpu=gpu,
            pre_transform=pre_transform,
            post_transform=post_transform
        )
        self.epsilon = epsilon

    def __str__(self):
        return 'IEnKSBundle(epsilon={0}, tau={1})'.format(
            str(self.epsilon.item()), str(self.tau.item())
        )

    def __repr__(self):
        return 'IEnKSBundle({0},{1})'.format(
            repr(self.epsilon.item()), repr(self.tau.item())
        )

    @property
    def epsilon(self):
        return self._core_module.epsilon

    @epsilon.setter
    @ensure_tensor
    @bound_tensor(min_val=0.0, max_val=None)
    def epsilon(self, new_epsilon):
        self._core_module = IEnKSBundleModule(
            epsilon=new_epsilon, tau=self._core_module.tau
        )

    @property
    def tau(self):
        return self._core_module.tau

    @tau.setter
    @ensure_tensor
    @bound_tensor(min_val=0.0, max_val=1.0)
    def tau(self, new_tau):
        self._core_module = IEnKSBundleModule(
            epsilon=self._core_module.epsilon, tau=new_tau
        )

    def get_model_weights(self, weights: xr.DataArray) -> xr.DataArray:
        weights_mean = weights.mean(dim='ensemble_new')
        prior_weights = self.generate_prior_weights(
            len(weights_mean['ensemble'])
        )
        epsilon_weights = self.epsilon.cpu().detach().numpy() * prior_weights
        epsilon_weights = epsilon_weights + weights_mean
        return epsilon_weights
