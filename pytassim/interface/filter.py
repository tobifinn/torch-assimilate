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
from typing import Union, Iterable, Tuple, List

# External modules
import xarray as xr
import pandas as pd

# Internal modules
from .base import BaseAssimilation


logger = logging.getLogger(__name__)


class FilterAssimilation(BaseAssimilation):
    @staticmethod
    def _slice_analysis(
            analysis_time: pd.Timestamp,
            state: xr.DataArray,
            observations: Iterable[xr.Dataset],
            pseudo_state: xr.DataArray,
    ) -> Tuple[xr.DataArray, xr.DataArray, Iterable[xr.Dataset]]:
        logger.info('Assimilation in filtering mode')
        state = state.sel(time=[analysis_time, ])
        pseudo_state = pseudo_state.sel(time=[analysis_time, ])
        sel_obs = []
        for obs in observations:
            tmp_obs = obs.sel(time=[analysis_time, ])
            tmp_obs.obs.operator = obs.obs.operator
            sel_obs.append(tmp_obs)
        observations = sel_obs
        return state, pseudo_state, observations

    @abc.abstractmethod
    def _get_weights(
            self,
            state: xr.DataArray,
            filtered_obs: List[xr.DataArray],
            ens_obs: List[xr.DataArray]
    ) -> xr.DataArray:
        pass

    def update_state(
            self,
            state: xr.DataArray,
            observations: Iterable[xr.Dataset],
            pseudo_state: Union[xr.DataArray, None],
            analysis_time: pd.Timestamp
    ) -> xr.DataArray:
        if pseudo_state is None:
            pseudo_state = state
        self._validate_state(pseudo_state)

        if not self.smoother:
            state, observations, pseudo_state = self._slice_analysis(
                analysis_time, state, observations, pseudo_state
            )

        ens_obs, filtered_obs = self._apply_obs_operator(pseudo_state,
                                                         observations)
        weights = self._get_weights(state, filtered_obs, ens_obs)
        analysis = self._apply_weights(state, weights)
        return analysis
