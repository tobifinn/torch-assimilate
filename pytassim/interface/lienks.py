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


class LocalizedIEnKSTransform(DomainLocalizedMixin, IEnKSTransform):
    def __init__(
            self,
            forward_model: Callable,
            localization: Union[None, BaseLocalization] = None,
            tau: int = 1.0,
            max_iter: int = 10,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None,
            chunksize: int = 10,
            weight_save_path: Union[None, str] = None
    ):
        super().__init__(
            forward_model=forward_model,
            tau=tau,
            max_iter=max_iter,
            smoother=smoother,
            gpu=gpu,
            pre_transform=pre_transform,
            post_transform=post_transform,
            weight_save_path=weight_save_path
        )
        self.localization = localization
        self.chunksize = chunksize

    def __str__(self):
        return 'Localized IEnKSTransform(loc={0}, tau={1})'.format(
            str(self.localization), str(self.tau.item())
        )

    def __repr__(self):
        return 'LIEnKSTransform({0},{1})'.format(
            repr(self.localization), repr(self.tau.item())
        )

    def inner_loop(
            self,
            state: xr.DataArray,
            weights: xr.DataArray,
            filtered_obs: List[xr.Dataset],
            ens_obs: List[xr.DataArray],
    ) -> xr.DataArray:
        innovations, ens_obs_perts = self._get_obs_space_variables(
            ens_obs, filtered_obs
        )
        logger.info('Got normalized data in observational space')

        obs_info = self._extract_obs_information(innovations)
        logger.info('Extracted observation grid information')
        logger.debug('Obs info: {0}'.format(obs_info))
        grid_index, state_info = self._extract_state_information(state)
        logger.info('Extracted grid information about the state id')
        logger.debug('State_id: {0}'.format(state_info))
        state_info = state_info.chunk({'grid': self.chunksize})
        logger.info('Chunked the state information')
        if 'grid' in weights.dims:
            weights['grid'] = state_info['grid']
            logger.debug('Weight_grid: {0}'.format(weights['grid']))
            logger.info('Removed grid information from weights')

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
        logger.info('Estimated the weights')
        weights = weights.assign_coords(grid=grid_index)
        weights['ensemble_new'] = weights.indexes['ensemble'].values
        logger.info('Post-processed the weights')
        return weights


class LocalizedIEnKSBundle(DomainLocalizedMixin, IEnKSBundle):
    inner_loop = LocalizedIEnKSTransform.inner_loop

    def __init__(
            self,
            forward_model: Callable,
            localization: Union[None, BaseLocalization] = None,
            tau: int = 1.0,
            epsilon: int = 1E-4,
            max_iter: int = 10,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None,
            chunksize: int = 10,
            weight_save_path: Union[None, str] = None
    ):
        super().__init__(
            forward_model=forward_model,
            tau=tau,
            epsilon=epsilon,
            max_iter=max_iter,
            smoother=smoother,
            gpu=gpu,
            pre_transform=pre_transform,
            post_transform=post_transform,
            weight_save_path=weight_save_path
        )
        self.localization = localization
        self.chunksize = chunksize

    def __str__(self):
        return 'Localized IEnKSBundle(loc={0}, eps={1}, tau={2})'.format(
            str(self.localization), str(self.epsilon.item()),
            str(self.tau.item())
        )

    def __repr__(self):
        return 'LIEnKSBundle({0},{1},{2})'.format(
            repr(self.localization), repr(self.epsilon.item()),
            repr(self.tau.item())
        )

