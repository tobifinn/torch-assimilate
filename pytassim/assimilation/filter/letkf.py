#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12/7/18
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

# External modules
import xarray as xr
import torch

# Internal modules
from .etkf import ETKFilter


logger = logging.getLogger(__name__)


class LETKFilter(ETKFilter):
    """
    This is an implementation of the `localized ensemble transform Kalman
    filter` [H07]_, which is a localized version of the `ensemble transform
    Kalman filter` [B01]_. This method iterates independently over each grid
    point in given background state. Given localization instance can be used to
    constrain the influence of observations in space. The ensemble weights are
    calculated for every grid point and independently applied to every grid
    point. This implementation follows [H07]_ with local weight estimation and
    is implemented in PyTorch. This implementation allows filtering in time
    based on linear propagation assumption [H04]_ and ensemble smoothing.

    References
    ----------
    .. [B01] Bishop, C. H., Etherton, B. J., & Majumdar, S. J. (2001).
             Adaptive sampling with the ensemble transform Kalman filter.
             Part I: Theoretical aspects. Monthly Weather Review, 129(3),
             420–436.
    .. [H04] Hunt, B., et al. Four-dimensional ensemble Kalman filtering.
             Tellus A, 56(4), 273–277.
    .. [H07] Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007).
             Efficient data assimilation for spatiotemporal chaos: A local
             ensemble transform Kalman filter. Physica D: Nonlinear
             Phenomena, 230(1), 112–126.
             https://doi.org/10.1016/j.physd.2006.11.008

    Parameters
    ----------
    smoothing : bool, optional
        Indicates if this filter should be run in smoothing or in filtering
        mode. In smoothing mode, no analysis time is selected from given state
        and the ensemble weights are applied to the whole state. In filtering
        mode, the weights are applied only on selected analysis time. Default
        is False, indicating filtering mode.
    localization : obj or None, optional
        This localization is used to localize and constrain observations
        spatially. If this localization is None, no localization is applied such
        it is an inefficient version of the `ensemble transform Kalman filter`.
        Default value is None, indicating no localization at all.
    """
    def __init__(self, smoothing=False, localization=None):
        super().__init__(smoothing=smoothing)
        self.localization = localization

    def update_state(self, state, observations, analysis_time):
        """
        This method updates the state based on given observations and analysis
        time. This method prepares different states, localize these states,
        iterates over state grid points, calculates the ensemble  weights and
        applies these weight to localized state. The calculation of the
        weights is based on PyTorch, while everything else is calculated with
        Numpy / Xarray.

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
        analysis_time : :py:class:`datetime.datetime`
            This analysis time determines at which point the state is updated.

        Returns
        -------
        analysis : :py:class:`xarray.DataArray`
            The analysed state based on given state and observations. The
            analysis has same coordinates as given ``state``. If filtering mode
            is on, then the time axis has only one element.
        """
        prepared_states = self._prepare(state, observations)
        if not self.smoothing:
            back_state = state.sel(time=[analysis_time, ])
        else:
            back_state = state
        analysis = []
        for grid_ind in state.grid.values:
            prepared_l = self._localize(prepared_states)
            prepared_l = [torch.tensor(s) for s in prepared_l]
            w_mean_l, w_perts_l = self._gen_weights(*prepared_l)
            back_state_l = back_state.sel(grid=grid_ind)
            state_mean_l, state_perts_l = back_state_l.state.split_mean_perts()
            ana_l = self._apply_weights(w_mean_l, w_perts_l, state_mean_l,
                                        state_perts_l)
            analysis.append(ana_l)
        analysis = xr.concat(analysis, dim='grid')
        analysis['grid'] = back_state['grid']
        analysis = analysis.transpose('var_name', 'time', 'ensemble', 'grid')
        return analysis

    def _localize(self, prepared_states):
        try:
            prepared_l = self.localization.localize_obs(prepared_states)
        except (NotImplementedError, AttributeError):
            prepared_l = prepared_states[:-1]
        return prepared_l
