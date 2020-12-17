#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 14.03.18
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2018}  {Tobias Sebastian Finn}
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# System modules
import logging
import abc
import warnings
import time
import datetime
from typing import Union, Iterable, Tuple, Any, List

# External modules
import xarray as xr
import pandas as pd
import torch
import numpy as np

# Internal modules
from .utils import grid_to_array

from pytassim.state import StateError
from pytassim.observation import ObservationError
from pytassim.transform import BaseTransformer


logger = logging.getLogger(__name__)


class BaseAssimilation(object):
    """
    BaseAssimilation is used as base class for all assimilation algorithms for
    fast prototyping of assimilation prototyping. To implement a data
    assimilation, one needs to overwrite
    :py:meth:`~pytassim.assimilation.base.BaseAssimilation.update_state`.
    """
    def __init__(
            self,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None
    ):
        self._dtype = torch.float32
        self.smoother = smoother
        self.gpu = gpu
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.dtype = torch.float64

    def __str__(self):
        return 'BaseAssimilation'

    def __repr__(self):
        return 'BaseAssimilation'

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, new_type):
        if isinstance(new_type, torch.dtype):
            self._dtype = new_type
        else:
            raise TypeError(
                'Given object is not a valid torch.dtype, '
                'instead it has as type: {0}'.format(type(new_type))
            )

    @property
    def _device(self) -> torch.device:
        return torch.device("cuda" if self.gpu else "cpu")

    @staticmethod
    def _validate_state(state: xr.DataArray):
        if not isinstance(state, xr.DataArray):
            raise TypeError(
                '*** Given state is not a valid ``xarray.DataArray`` ***\n'
                '{0}'.format(type(state))
            )
        if not state.state.valid:
            err_msg = '*** Given state is not a valid state ***\n{0:s}'
            raise StateError(err_msg.format(str(state)))

    @staticmethod
    def _validate_observations(observations: Iterable[xr.Dataset]):
        for obs in observations:
            if not isinstance(obs, xr.Dataset):
                raise TypeError(
                    '*** Given observation is not a valid ``xarray.Dataset`` '
                    '***\n{0:s}'.format(obs)
                )
            if not obs.obs.valid:
                raise ObservationError(
                    '*** Given observation is not a valid observation ***\n'
                    '{0:s}'.format(str(obs))
                )

    @staticmethod
    def _get_analysis_time(
            state: xr.DataArray,
            analysis_time: Any = None
    ) -> pd.Timestamp:
        if analysis_time is None:
            valid_time = state.time[-1]
        else:
            analysis_time = pd.to_datetime(analysis_time)
            try:
                valid_time = state.time.sel(time=analysis_time, method=None)
            except KeyError:
                valid_time = state.time.sel(time=analysis_time,
                                            method='nearest')
                warnings.warn(
                    'Given analysis time {0:s} is not within state, used '
                    'instead nearest neighbor {1:s}'.format(
                        str(analysis_time), str(valid_time.values)
                    ),
                    category=UserWarning
                )
        valid_time = valid_time.values
        valid_time = pd.to_datetime(valid_time)
        return valid_time

    @staticmethod
    def _apply_obs_operator(
            pseudo_state: xr.DataArray,
            observations: Iterable[xr.Dataset]
    ) -> Tuple[List[xr.DataArray], List[xr.Dataset]]:
        """
        This method applies the observation operator on given state. The
        observation operator has to be set within given observations. It is
        possible to overwrite this method to implement an own observation
        operator, which was not set in given observations.

        Parameters
        ----------
        pseudo_state : :py:class:`xarray.DataArray`
            This state is used as base state to apply set observation operator.
        observations : iterable(:py:class:`xarray.Dataset`)
            These observations are used as basis for the observation operators.
            The observation operator should be set as method
            :py:meth:`xarray.Dataset.obs.operator`.

        Returns
        -------
        obs_equivalent : list(:py:class:`xarray.DataArray`)
            A list with observation equivalents as :py:class:`xarray.DataArray`.
            These observation equivalents have three dimensions, ``ensemble``,
            ``time`` and ``obs_grid``. The order within these observation
            equivalent is the same as in the observations.
        filtered_observations : list(:py:class:`xarray.Dataset`)
            These observations are filtered such that observations without an
            observation operator are dropped.
        """
        obs_equivalent = []
        filtered_observations = []
        for obs in observations:
            try:
                obs_equivalent.append(obs.obs.operator(obs, pseudo_state))
                filtered_observations.append(obs)
            except NotImplementedError:
                pass
        return obs_equivalent, filtered_observations

    @staticmethod
    def _stack_obs(
            observations: List[xr.DataArray]
    ) -> xr.DataArray:
        stacked_observations = []
        for obs in observations:
            if isinstance(obs.indexes['obs_grid_1'], pd.MultiIndex):
                obs['obs_grid_1'] = pd.Index(
                    obs.indexes['obs_grid_1'].values, tupleize_cols=False
                )
            stacked_obs = obs['observations'].stack(
                obs_id=('time', 'obs_grid_1')
            )
            stacked_observations.append(stacked_obs)
        stacked_observations = xr.concat(stacked_observations, dim='obs_id')
        return stacked_observations

    @staticmethod
    def _apply_weights(
            state: xr.DataArray,
            weights: xr.DataArray
    ) -> xr.DataArray:
        state_mean, state_perts = state.state.split_mean_perts(dim='ensemble')
        analysis_perts = xr.dot(state_perts, weights, dims='ensemble')
        analysis_perts = analysis_perts.rename({'ensemble_new': 'ensemble'})
        analysis = state_mean + analysis_perts
        return analysis

    @abc.abstractmethod
    def update_state(
            self,
            state: xr.DataArray,
            observations: Iterable[xr.Dataset],
            pseudo_state: Union[xr.DataArray, None],
            analysis_time: pd.Timestamp
    ) -> xr.DataArray:
        """
        This method is called by
        :py:meth:`~pytassim.assimilation.base.BaseAssimilation.assimilate` and
        has to be overwritten to implement a data assimilation algorithm.

        Parameters
        ----------
        state : :py:class:`xarray.DataArray`
            This state is updated by this assimilation algorithm and given
            ``observation``. This :py:class:`~xarray.DataArray` should have
            four coordinates, which are specified in
            :py:class:`pytassim.state.ModelState`.
        observations : iterable(:py:class:`xarray.Dataset`)
            These observations are used to update given state.
        pseudo_state : :py:class:`xarray.DataArray`
            This state is used to generate an observation-equivalent. This
             :py:class:`~xarray.DataArray` should have four coordinates, which
             are specified in :py:class:`pytassim.state.ModelState`.
        analysis_time : :py:class:`datetime.datetime`
            This analysis time determines at which point the state is updated.

        Returns
        -------
        analysis : :py:class:`xarray.DataArray`
            The analysed state based on given state and observations. The
            analysis has same coordinates as given ``state`` except ``time``,
            which should contain only one time step.
        """
        pass

    def assimilate(
            self,
            state: xr.DataArray,
            observations: Union[xr.Dataset, Iterable[xr.Dataset]],
            pseudo_state: Union[xr.DataArray, None] = None,
            analysis_time: Any = None
    ) -> xr.DataArray:
        """
        This assimilate the ``observations`` in given background ``state`` and
        creates an analysis for given ``analysis_time``. The observations need
        an observation operator, which translate given state into an
        observation-equivalent. The state, observations, observation covariance
        and and the observation-equivalent are used to update given state. This
        method validates given state and observations, gets analysis time and
        calls
        :py:meth:`~pytassim.assimilation.base.BaseAssimilation.update_state`.
        If state to assimilate is given, this state is translated into
        observation space.

        Parameters
        ----------
        state : :py:class:`xarray.DataArray`
            This state is updated by this assimilation algorithm and given
            ``observation``. This :py:class:`~xarray.DataArray` should have
            four coordinates, which are specified in
            :py:class:`pytassim.state.ModelState`. If no pseudo_state is
            specified, this state is also used to generate pseudo observations.
        observations : :py:class:`xarray.Dataset` or \
        iterable(:py:class:`xarray.Dataset`)
            These observations are used to update given state. An iterable of
            many :py:class:`xarray.Dataset` can be used to assimilate different
            variables. For the observation state, these observations are
            stacked such that the observation state contains all observations.
            The :py:class:`xarray.Dataset` are validated with
            :py:class:`pytassim.observation.Observation.valid`
        pseudo_state : :py:class:`xarray.DataArray` or None
            If this additional state is given, this state is used to create
            pseudo-observations. This :py:class:`~xarray.DataArray` should have
            four coordinates, which are specified in
            :py:class:`pytassim.state.ModelState`.
        analysis_time : :py:class:`datetime.datetime` or None, optional
            This analysis time determines at which point the state is updated.
            If the analysis time is None, than the last time point in given
            state is used.

        Returns
        -------
        analysis : :py:class:`xarray.DataArray`
            The analysed state based on given state and observations. The
            analysis has same coordinates as given ``state`` except ``time``,
            which contains only one time step.
        """
        start_time = time.time()
        logger.info('Starting assimilation')
        if not observations:
            warnings.warn('No observation is given, I will return the '
                          'background state!', UserWarning)
            return state
        if not isinstance(observations, (list, set, tuple)):
            observations = (observations, )
        self._validate_state(state)
        self._validate_observations(observations)
        analysis_time = self._get_analysis_time(state, analysis_time)
        if isinstance(analysis_time, datetime.datetime):
            logger.info(
                'Analysis time: {0:s}'.format(
                    analysis_time.strftime('%Y-%m-%d %H:%M UTC')
                )
            )
        if self.pre_transform:
            for trans in self.pre_transform:
                back_state, observations, pseudo_state = trans.pre(
                    state, observations, pseudo_state
                )
        logger.info('Finished with general preparation')
        analysis = self.update_state(
            state, observations, pseudo_state, analysis_time
        )
        logger.info('Created the analysis, starting with post-processing')
        if self.post_transform:
            for trans in self.post_transform:
                analysis = trans.post(analysis, state, observations,
                                      pseudo_state)
        self._validate_state(analysis)
        end_time = time.time()
        logger.info('Finished assimilation after {0:.2f} s'.format(
            end_time-start_time
        ))
        return analysis
