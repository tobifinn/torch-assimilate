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

# External modules
import xarray as xr

# Internal modules
from pytassim.state import StateError
from pytassim.observation import ObservationError


logger = logging.getLogger(__name__)


class BaseAssimilation(object):
    def __init__(self):
        pass

    @staticmethod
    def _validate_state(state):
        if not isinstance(state, xr.DataArray):
            raise TypeError('*** Given state is not a valid '
                            '``xarray.DataArray`` ***\n{0:s}'.format(state))
        if not state.state.valid:
            err_msg = '*** Given state is not a valid state ***\n{0:s}'
            raise StateError(err_msg.format(str(state)))

    @staticmethod
    def _validate_single_obs(observation):
        if not isinstance(observation, xr.Dataset):
            raise TypeError('*** Given observation is not a valid'
                            '``xarray.Dataset`` ***\n{0:s}'.format(observation))
        if not observation.obs.valid:
            err_msg = '*** Given observation is not a valid observation ***' \
                      '\n{0:s}'
            raise ObservationError(err_msg.format(str(observation)))

    def _validate_observations(self, observations):
        if isinstance(observations, (list, set, tuple)):
            for obs in observations:
                self._validate_single_obs(obs)
        else:
            self._validate_single_obs(observations)

    @staticmethod
    def _get_analysis_time(state, analysis_time=None):
        if analysis_time is None:
            valid_time = state.time[-1]
        else:
            try:
                valid_time = state.time.sel(time=analysis_time, method=None)
            except KeyError:
                valid_time = state.time.sel(time=analysis_time,
                                            method='nearest')
                warnings.warn(
                    'Given analysis time {0:s} is not within state, used '
                    'instead nearest neighbor {1:s}'.format(
                        str(analysis_time), str(valid_time)
                    ),
                    category=UserWarning
                )
        return valid_time

    @abc.abstractmethod
    def update_state(self, state, observations, analysis_time=None):
        pass

    def assimilate(self, state, observations, analysis_time=None):
        """
        This assimilate the ``observations`` in given background ``state`` and
        creates an analysis for given ``analysis_time``. The observations need
        an observation operator, which translate given state into an
        observation-equivalent. The state, observations, observation covariance
        and and the observation-equivalent are used to update given state.

        Parameters
        ----------
        state : :py:class:`xarray.DataArray`
            This state is used to generate an observation-equivalent. It is
            further updated by this assimilation algorithm and given
            ``observation``. This :py:class:`~xarray.DataArray` should have
            four coordinates, which are specified in
            :py:class:`pytassim.state.ModelState`.
        observations : :py:class:`xarray.Dataset` or \
        iterable(:py:class:`xarray.Dataset`)
            These observations are used to update given state. An iterable of
            many :py:class:`xarray.Dataset` can be used to assimilate different
            variables. For the observation state, these observations are
            stacked such that the observation state contains all observations.
            The :py:class:`xarray.Dataset` are validated with
            :py:class:`pytassim.observation.Observation.valid`
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
        pass
