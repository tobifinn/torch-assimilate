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
    torch_innovations = torch.from_numpy(innovations).to(device=device,
                                                         dtype=dtype)
    torch_perts = torch.from_numpy(ens_obs_perts).to(device=device, dtype=dtype)
    torch_innovations = torch_innovations.view(1, obs_size)
    torch_perts = torch_perts.view(ens_size, obs_size)
    torch_weights = core_module(torch_perts, torch_innovations)[0]
    torch_weights = torch_weights.cpu().detach()
    weights = torch_weights.numpy().astype(innovations.dtype)
    return weights


class ETKF(FilterAssimilation):
    """
    This is an implementation of the `ensemble transform Kalman filter`
    :cite:`bishop_adaptive_2001`.
    This ensemble Kalman filter is a deterministic filter, where the state is
    update globally. This ensemble Kalman filter estimates ensemble weights in
    weight space, which are then applied to the given state. This implementation
    follows :cite:`hunt_efficient_2007` with global weight estimation and is
    implemented in PyTorch.
    This implementation allows filtering in time based on linear propagation
    assumption :cite:`hunt_four-dimensional_2004` and ensemble smoothing.

    Parameters
    ----------
    smoother : bool, optional
        Indicates if this filter should be run in smoothing or in filtering
        mode. In smoothing mode, no analysis time is selected from given state
        and the ensemble weights are applied to the whole state. In filtering
        mode, the weights are applied only on selected analysis time. Default
        is False, indicating filtering mode.
    inf_factor : float, optional
        Multiplicative inflation factor :math:`\\rho``, which is applied to the
        background precision. An inflation factor greater one increases the
        ensemble spread, while a factor less one decreases the spread. Default
        is 1.0, which is the same as no inflation at all.
    gpu : bool, optional
        Indicator if the weight estimation should be done on either GPU (True)
        or CPU (False): Default is None. For small models, estimation of the
        weights on CPU is faster than on GPU!.
    """
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
        self.inf_factor = inf_factor

    def __str__(self):
        return 'Global ETKF(rho={0})'.format(str(self.inf_factor))

    def __repr__(self):
        return 'ETKF({0})'.format(repr(self.inf_factor))

    @property
    def module(self):
        return self._core_module

    @property
    def inf_factor(self):
        return self._core_module.inf_factor

    @inf_factor.setter
    def inf_factor(self, new_factor):
        if isinstance(new_factor, (float, int)):
            new_factor = torch.tensor(new_factor, dtype=self.dtype)
        self._core_module = ETKFModule(inf_factor=new_factor)

    def estimate_weights(
            self,
            state: xr.DataArray,
            filtered_obs: List[xr.Dataset],
            ens_obs: List[xr.DataArray]
    ) -> xr.DataArray:
        innovations, ens_obs_perts = self._get_obs_space_variables(
            ens_obs, filtered_obs
        )

        weights = xr.apply_ufunc(
            etkf_function,
            innovations,
            ens_obs_perts,
            input_core_dims=[['obs_id'], ['ensemble', 'obs_id']],
            dask='parallelized',
            output_core_dims=[['ensemble', 'ensemble_new']],
            output_dtypes=[float],
            output_sizes={'ensemble_new': len(ens_obs_perts['ensemble'])},
            kwargs={
                'core_module': self._core_module,
                'device': self.device,
                'dtype': self.dtype
            }
        )
        return weights
