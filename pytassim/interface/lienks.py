#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 04.01.21
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union, Callable, Iterable, List

# External modules
import torch
import xarray as xr

# Internal modules
from .ienks import IEnKSTransform, IEnKSBundle
from .mixin_local import DomainLocalizedMixin
from pytassim.localization.localization import BaseLocalization
from pytassim.transform.base import BaseTransformer


logger = logging.getLogger(__name__)


class LocalizedIEnKSTransform(IEnKSTransform, DomainLocalizedMixin):
    def __init__(
            self,
            model: Callable,
            localization: Union[None, BaseLocalization] = None,
            tau: int = 1.0,
            max_iter: int = 10,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None,
            chunksize: int = 10,
    ):
        super().__init__(
            model=model,
            tau=tau,
            max_iter=max_iter,
            smoother=smoother,
            gpu=gpu,
            pre_transform=pre_transform,
            post_transform=post_transform
        )
        self.localization = localization
        self.chunksize = chunksize

    def __str__(self):
        return 'Localized IEnKSBundle(loc={0}, tau={1})'.format(
            str(self.localization), str(self.tau)
        )

    def __repr__(self):
        return 'LIEnKSBundle({0},{1})'.format(
            repr(self.localization), repr(self.tau)
        )

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
        obs_info = self._extract_obs_information(innovations)
        state_index, state_info = self._extract_state_information(state)
        state_info = state_info.chunk({'state_id': self.chunksize})
        weights = self._weights_stack_state_id(weights)

        self._core_module = torch.jit.script(self._core_module)

        weights = xr.apply_ufunc(
            self.localized_module,
            state_info,
            weights,
            ens_obs_perts,
            innovations,
            input_core_dims=[
                ['id_names'],
                ['ensemble', 'ensemble_new'],
                ['ensemble', 'obs_id'],
                ['obs_id']
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[['ensemble', 'ensemble_new']],
            output_dtypes=[float],
            kwargs={
                'obs_info': obs_info,
                'args_to_skip': (0, )
            }
        )
        weights = weights.assign_coords(state_id=state_index)
        weights = weights.unstack('state_id')
        weights['time'] = state.indexes['time']
        return weights


class LocalizedIEnKSBundle(IEnKSBundle, DomainLocalizedMixin):
    estimate_weights = LocalizedIEnKSTransform.estimate_weights

    def __init__(
            self,
            model: Callable,
            localization: Union[None, BaseLocalization] = None,
            tau: int = 1.0,
            epsilon: int = 1E-4,
            max_iter: int = 10,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None,
            chunksize: int = 10,
    ):
        super().__init__(
            model=model,
            tau=tau,
            epsilon=epsilon,
            max_iter=max_iter,
            smoother=smoother,
            gpu=gpu,
            pre_transform=pre_transform,
            post_transform=post_transform
        )
        self.localization = localization
        self.chunksize = chunksize

    def __str__(self):
        return 'Localized IEnKSBundle(loc={0}, eps={1}, tau={2})'.format(
            str(self.localization), str(self.epsilon), str(self.tau)
        )

    def __repr__(self):
        return 'LIEnKSBundle({0},{1},{2})'.format(
            repr(self.localization), repr(self.epsilon), repr(self.tau)
        )

