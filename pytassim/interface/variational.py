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
            forward_model: Callable,
            max_iter: int = 10,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None,
            weight_save_path: Union[None, str] = None,
    ):
        super().__init__(
            smoother=smoother,
            gpu=gpu,
            pre_transform=pre_transform,
            post_transform=post_transform,
            forward_model=forward_model,
            weight_save_path=weight_save_path
        )
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
    def inner_loop(
            self,
            state: xr.DataArray,
            weights: xr.DataArray,
            observations: List[xr.Dataset],
            ens_obs: List[xr.DataArray],
    ) -> xr.DataArray:
        pass

    def _outer_step(
            self,
            weights: xr.DataArray,
            state: xr.DataArray,
            observations: Iterable[xr.Dataset],
            pseudo_state: Union[xr.DataArray, None],
            iter_num: int = 0
    ) -> xr.DataArray:
        pseudo_state = self.get_pseudo_state(
            pseudo_state=pseudo_state,
            state=state,
            weights=weights,
            iter_num=iter_num
        )
        ens_obs, filtered_obs = self._apply_obs_operator(
            pseudo_state, observations
        )
        weights = self.inner_loop(state, weights, filtered_obs, ens_obs)
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
            weights = self._outer_step(
                weights=weights,
                state=state,
                observations=observations,
                pseudo_state=pseudo_state,
                iter_num=iter_num
            )
            weights = self.precompute_weights(weights)
            pseudo_state = None
            iter_num += 1
        analysis_state = self._apply_weights(state, weights)
        if self.smoother:
            analysis_state, _ = self.forward_model(analysis_state, iter_num)
        return analysis_state
