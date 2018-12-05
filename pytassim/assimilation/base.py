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

# External modules

# Internal modules


logger = logging.getLogger(__name__)


class BaseAssimilation(object):
    def __init__(self):
        pass

    @abc.abstractmethod
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
