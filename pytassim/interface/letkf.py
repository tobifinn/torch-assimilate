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
from .etkf import ETKF, etkf_function
from .mixin_local import DomainLocalizedMixin
from ..transform.base import BaseTransformer
from ..localization.localization import BaseLocalization


logger = logging.getLogger(__name__)


class LETKF(ETKF, DomainLocalizedMixin):
    def __init__(
            self,
            localization: Union[None, BaseLocalization] = None,
            inf_factor: Union[float, torch.Tensor, torch.nn.Parameter] = 1.0,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None,
            chunksize: int = 10,
    ):
        super().__init__(inf_factor=inf_factor, smoother=smoother, gpu=gpu,
                         pre_transform=pre_transform,
                         post_transform=post_transform)
        self.localization = localization
        self.chunksize = chunksize

    def __str__(self):
        return 'Localized ETKF(gamma={0}, loc={1})'.format(
            str(self.inf_factor), str(self.localization)
        )

    def __repr__(self):
        return 'LETKF({0},{1})'.format(
            repr(self.inf_factor), repr(self.localization)
        )

    def estimate_weights(
            self,
            state: xr.DataArray,
            filtered_obs: List[xr.Dataset],
            ens_obs: List[xr.DataArray]
    ) -> xr.DataArray:
        innovations, ens_obs_perts = self._get_obs_space_variables(
            ens_obs, filtered_obs
        )

        obs_info = self._extract_obs_information(innovations)
        state_index, state_info = self._extract_state_information(state)
        state_info = state_info.chunk({'state_id': self.chunksize})
        localized_etkf_function = self.localization_decorator(etkf_function)

        weights = xr.apply_ufunc(
            localized_etkf_function,
            state_info,
            innovations,
            ens_obs_perts,
            input_core_dims=[['id_names'], ['obs_id'], ['ensemble', 'obs_id']],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[['ensemble', 'ensemble_new']],
            output_dtypes=[float],
            output_sizes={'ensemble_new': len(state['ensemble'])},
            kwargs={
                'obs_info': obs_info,
                'core_module': self._core_module,
                'device': self.device,
                'dtype': self.dtype
            }
        )
        weights = weights.assign_coords(state_id=state_index)
        weights = weights.unstack('state_id')
        weights['time'] = state.indexes['time']
        return weights
