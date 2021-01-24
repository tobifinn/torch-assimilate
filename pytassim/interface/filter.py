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
    """
    This is an AbstractClass for filtering-based data assimilation
    techniques, like ensemble Kalman filters or particle filters.
    To initiate a filtering-based technique, one only needs to overwrite the
    :py:meth:`~pytassim.interface.filter.FilterAssimilation
    .estimate_weights` method with a method, which returns the estimated
    weights.
    """
    @staticmethod
    def _slice_analysis(
            analysis_time: pd.Timestamp,
            state: xr.DataArray,
            observations: Iterable[xr.Dataset],
            pseudo_state: xr.DataArray,
    ) -> Tuple[xr.DataArray, Iterable[xr.Dataset], xr.DataArray]:
        logger.info('Assimilation in filtering mode')
        state = state.sel(time=[analysis_time, ])
        pseudo_state = pseudo_state.sel(time=[analysis_time, ])
        sel_obs = []
        for obs in observations:
            tmp_obs = obs.sel(time=[analysis_time, ])
            tmp_obs.obs.operator = obs.obs.operator
            sel_obs.append(tmp_obs)
        observations = sel_obs
        return state, observations, pseudo_state

    @abc.abstractmethod
    def estimate_weights(
            self,
            state: xr.DataArray,
            filtered_obs: List[xr.DataArray],
            ens_obs: List[xr.DataArray]
    ) -> xr.DataArray:
        """
        This method is used to estimate the weights with given state,
        filtered observations and the ensemble equivalent of the observations.

        Parameters
        ----------
        state : :py:class:`xarray.DataArray`
            This state is updated by this assimilation algorithm and given
            ``observation``. This :py:class:`~xarray.DataArray` should have
            four coordinates, which are specified in
            :py:class:`pytassim.state.ModelState`.
        filtered_obs : list(:py:class:`xarray.Dataset`)
            These observations are used to update given state. An iterable of
            many :py:class:`xarray.Dataset` can be used to assimilate different
            variables. For the observation state, these observations are
            stacked such that the observation state contains all observations.
        ens_obs : list(:py:class:`xarray.DataArray`)
            The ensemble equivalent of the observations. This list should
            have the same length as the `filtered_obs` list.

        Returns
        -------
        weights : :py:class:`xarray.DataArray`
            The estimated ensemble weights based on given state, filtered
            observations and ensemble observations. The weights should have
            at least "ensemble" and "ensemble_new" as dimensions. The
            `ensemble` dimension should be the same as the `ensemble`
            dimension of the given `state`. `ensemble_new` could have another
            length if the number of ensemble members should be changed witihn
            the analysis.
        """
        pass

    def update_state(
            self,
            state: xr.DataArray,
            observations: Iterable[xr.Dataset],
            pseudo_state: Union[xr.DataArray, None],
            analysis_time: pd.Timestamp
    ) -> xr.DataArray:
        """
        This method updates the state based on given observations and analysis
        time. This method prepares the different states, calculates the ensemble
        weights and applies these weight to given state. The calculation of the
        weights is based on PyTorch, while everything else is calculated with
        Numpy / Xarray.

        Parameters
        ----------
        state : :py:class:`xarray.DataArray`
            This state is updated by this assimilation algorithm and given
            ``observation``. This :py:class:`~xarray.DataArray` should have
            four coordinates, which are specified in
            :py:class:`pytassim.state.ModelState`.
        observations : list(:py:class:`xarray.Dataset`)
            These observations are used to update given state. An iterable of
            many :py:class:`xarray.Dataset` can be used to assimilate different
            variables. For the observation state, these observations are
            stacked such that the observation state contains all observations.
        pseudo_state : :py:class:`xarray.DataArray` or None, optional
            This state is used to generate an observation-equivalent. This
             :py:class:`~xarray.DataArray` should have four coordinates, which
             are specified in :py:class:`pytassim.state.ModelState`. Default
             is None.
        analysis_time : :py:class:`pd.TimeStamp`
            This analysis time determines at which point the state is updated.

        Returns
        -------
        analysis : :py:class:`xarray.DataArray`
            The analysed state based on given state and observations. The
            analysis has same coordinates as given ``state``, except
            `ensemble` which can change its size. If
            filtering mode is on, then the time axis is sliced to the
            analysis time.
        """
        prior_weights = self.generate_prior_weights(state.indexes['ensemble'])
        pseudo_state = self.get_pseudo_state(
            pseudo_state=pseudo_state,
            state=state,
            weights=prior_weights
        )
        self._validate_state(pseudo_state)

        if not self.smoother:
            state, observations, pseudo_state = self._slice_analysis(
                analysis_time, state, observations, pseudo_state
            )

        ens_obs, filtered_obs = self._apply_obs_operator(pseudo_state,
                                                         observations)
        logger.info('Start to estimate the weights')
        weights = self.estimate_weights(state, filtered_obs, ens_obs)
        logger.info('Finished with weight estimation, starting with '
                    'application of weights')
        analysis = self._apply_weights(state, weights)
        return analysis
