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
from .etkf import ETKF
from .mixin_local import DomainLocalizedMixin
from ..transform.base import BaseTransformer
from ..localization.localization import BaseLocalization


logger = logging.getLogger(__name__)


class LETKF(DomainLocalizedMixin, ETKF):
    """
    This is an implementation of the `localized ensemble transform Kalman
    filter` :cite:`hunt_efficient_2007`.
    This is a localized version of the `ensemble transform Kalman filter`
    :cite:`bishop_adaptive_2001`. This method iterates independently over each
    grid point in given background state. Given localization instance can be
    used to
    constrain the influence of observations in space. The ensemble weights are
    calculated for every grid point and independently applied to every grid
    point. This implementation follows :cite:`hunt_efficient_2007`, with local
    weight estimation and is implemented in PyTorch. This implementation allows
    filtering in time based on linear propagation assumption
    :cite:`hunt_four-dimensional_2004` and ensemble smoothing.

    Parameters
    ----------
    localization : obj or None, optional
        This localization is used to localize and constrain observations
        spatially. If this localization is None, no localization is applied such
        it is an inefficient version of the `ensemble transform Kalman filter`.
        Default value is None, indicating no localization at all.
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
        return 'Localized ETKF(rho={0}, loc={1})'.format(
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
        logger.info('Got normalized data in observational space')

        obs_info = self._extract_obs_information(innovations)
        state_index, state_info = self._extract_state_information(state)
        state_info = state_info.chunk({'state_id': self.chunksize})
        logger.info('Extracted grid information about the state id')
        logger.debug('State_id: {0}'.format(state_info))

        self._core_module = torch.jit.script(self._core_module)
        logger.info('Compiled the core module')

        weights = xr.apply_ufunc(
            self.localized_module,
            state_info,
            ens_obs_perts,
            innovations,
            input_core_dims=[['id_names'], ['ensemble', 'obs_id'], ['obs_id']],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[['ensemble', 'ensemble_new']],
            output_dtypes=[float],
            dask_gufunc_kwargs=dict(
                output_sizes={'ensemble_new': len(ens_obs_perts['ensemble'])}
            ),
            kwargs={
                'obs_info': obs_info,
            }
        )
        logger.info('Estimated the weights')
        weights = weights.assign_coords(state_id=state_index)
        weights = weights.unstack('state_id')
        weights['time'] = state.indexes['time']
        logger.info('Post-processed the weights')
        return weights
