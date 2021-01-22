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
from typing import Union, Iterable, Callable, List

# External modules
import xarray as xr
import pandas as pd
import torch
import numpy as np

# Internal modules
from pytassim.state import StateError
from pytassim.observation import ObservationError
from pytassim.transform import BaseTransformer
from .base import BaseAssimilation


logger = logging.getLogger(__name__)


class VarAssimilation(BaseAssimilation):
    def __init__(
            self,
            model: Callable,
            weight_save_path: Union[None, str] = None,
            max_iter: int = 10,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None,
    ):
        super().__init__(
            smoother=smoother,
            gpu=gpu,
            pre_transform=pre_transform,
            post_transform=post_transform
        )
        self.model = model
        self.max_iter = max_iter
        self.weight_save_path = weight_save_path

    def precompute_weights(self, weights: xr.DataArray) -> xr.DataArray:
        if isinstance(self.weight_save_path, str):
            weights.to_netcdf(self.weight_save_path)
            weights = xr.open_dataarray(
                self.weight_save_path, chunks=weights.chunks
            )
        else:
            weights = weights.load()
        return weights

    @abc.abstractmethod
    def get_model_weights(self, weights: xr.DataArray) -> xr.DataArray:
        pass

    @abc.abstractmethod
    def estimate_weights(
            self,
            state: xr.DataArray,
            weights: xr.DataArray,
            observations: List[xr.Dataset],
            ens_obs: List[xr.DataArray],
    ) -> xr.DataArray:
        pass

    @staticmethod
    def generate_prior_weights(ens_values: pd.Index) -> xr.DataArray:
        prior_weights = np.eye(len(ens_values))
        prior_weights = xr.DataArray(
            prior_weights,
            coords={
                'ensemble': ens_values,
                'ensemble_new': ens_values
            },
            dims=['ensemble', 'ensemble_new']
        )
        return prior_weights

    def _propagate_model(
            self,
            weights: xr.DataArray,
            state: xr.DataArray,
            iter_num: int = 0
    ) -> xr.DataArray:
        model_weights = self.get_model_weights(weights)
        model_state = self._apply_weights(state, model_weights)
        _, pseudo_state = self.model(
            state=model_state,
            iter_num=iter_num
        )
        self._validate_state(pseudo_state)
        return pseudo_state

    def _weights_stack_state_id(self, weights: xr.DataArray) -> xr.DataArray:
        if 'grid' in weights.dims:
            weights = weights.state.stack_to_state_id()
            weights['state_id'] = np.arange(len(weights['state_id']))
            weights = weights.chunk({'state_id': self.chunksize})
        return weights

    def _update_step(
            self,
            weights: xr.DataArray,
            state: xr.DataArray,
            observations: Iterable[xr.Dataset],
            pseudo_state: Union[xr.DataArray, None],
            iter_num: int = 0,
    ) -> xr.DataArray:
        if pseudo_state is None:
            pseudo_state = self._propagate_model(weights, state,
                                                 iter_num=iter_num)
        ens_obs, filtered_obs = self._apply_obs_operator(
            pseudo_state, observations
        )
        weights = self.estimate_weights(state, weights, filtered_obs, ens_obs)
        return weights

    def update_state(
            self,
            state: xr.DataArray,
            observations: Iterable[xr.Dataset],
            pseudo_state: Union[xr.DataArray, None],
            analysis_time: pd.Timestamp
    ) -> xr.DataArray:
        weights = self.generate_prior_weights(state.indexes['ensemble'])
        state = state.sel(time=[analysis_time])
        iter_num = 0
        while iter_num < self.max_iter:
            weights = self._update_step(weights, state, observations,
                                        pseudo_state, iter_num=iter_num)
            weights = self.precompute_weights(weights)
            pseudo_state = None
            iter_num += 1
        analysis_state = self._apply_weights(state, weights)
        if self.smoother:
            analysis_state, _ = self.model(analysis_state, iter_num)
        return analysis_state
