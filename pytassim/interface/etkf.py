#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 17.12.20
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}


# System modules
import logging
import abc
from typing import Union, Iterable, List, Callable

# External modules
import xarray as xr
import numpy as np
import torch
import torch.nn

# Internal modules
from .filter import FilterAssimilation
from ..core import ETKFModule
from ..transform.base import BaseTransformer


logger = logging.getLogger(__name__)


__all__ = ['ETKF']


def etkf_function(
        innovations: np.ndarray,
        ens_obs_perts: np.ndarray,
        core_module: Callable,
        device: torch.device,
        dtype: torch.dtype
) -> np.ndarray:
    obs_size = innovations.shape[0]
    ens_size = ens_obs_perts.shape[0]
    torch_innovations = torch.from_numpy(innovations).to(dtype).to(device)
    torch_perts = torch.from_numpy(ens_obs_perts).to(dtype).to(device)
    torch_innovations = torch_innovations.view(1, obs_size)
    torch_perts = torch_perts.view(ens_size, obs_size)
    torch_weights = core_module(torch_perts, torch_innovations)[0]
    weights = torch_weights.numpy().astype(innovations.dtype)
    return weights


class ETKF(FilterAssimilation):
    def __init__(
            self,
            inf_factor: Union[float, torch.Tensor, torch.nn.Parameter] = 1.0,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None
    ):
        super().__init__(smoother=smoother, gpu=gpu,
                         pre_transform=pre_transform,
                         post_transform=post_transform)
        self._core_module = None
        self.inf_factor = inf_factor

    def __str__(self):
        return 'Global ETKF({0})'.format(self.inf_factor)

    def __repr__(self):
        return 'ETKF'

    @property
    def module(self):
        return self._core_module

    @property
    def inf_factor(self):
        return self._core_module.inf_factor

    @inf_factor.setter
    def inf_factor(self, new_factor):
        self._core_module = ETKFModule(inf_factor=new_factor)

    @abc.abstractmethod
    def _estimate_weights(
            self,
            state: xr.DataArray,
            filtered_obs: List[xr.Dataset],
            ens_obs: List[xr.DataArray]
    ) -> xr.DataArray:
        innovations = []
        ens_obs_perts = []
        for k, curr_ens in enumerate(ens_obs):
            curr_mean, curr_perts = curr_ens.state.split_mean_perts(
                dim='ensemble'
            )
            curr_innov = filtered_obs[k]['observations']-curr_mean
            curr_innov = filtered_obs[k].obs.mul_rcinv(curr_innov)
            curr_perts = filtered_obs[k].obs.mul_rcinv(curr_perts)
            innovations.append(curr_innov)
            ens_obs_perts.append(curr_perts)
        innovations = self._stack_obs(innovations)
        ens_obs_perts = self._stack_obs(ens_obs_perts)

        weights = xr.apply_ufunc(
            etkf_function,
            innovations,
            ens_obs_perts,
            input_core_dims=[['obs_id'], ['ensemble', 'obs_id']],
            dask='parallelized',
            output_core_dims=[['ensemble', 'ensemble_new']],
            output_dtypes=[float],
            output_sizes={'ensemble_new': len(state['ensemble'])},
            kwargs={
                'core_module': self._core_module,
                'device': self.device,
                'dtype': self._dtype
            }
        )
        return weights
