#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12/6/18
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
import numpy as np
import torch

# Internal modules


logger = logging.getLogger(__name__)


def dummy_obs_operator(obs_ds, state):
    """
    This dummy observation operator can be used to patch
    :py:meth:`~pytassim.observation.Observation.operator` for testing purpose.
    This function / method select the `x` variable of given ``state`` and
    renames `grid` to `obs_grid_1`. `time` and `obs_grid_1` are further new set
    based on values of given observation instance in self.

    Parameters
    ----------
    obs_ds : :py:class:`~xarray.Dataset`
        This observational dataset is used as basis dataset.
    state : :py:class:`~xarray.DataArray`
        The pseudo observations are created based on this state.

    Returns
    -------
    pseudo_obs : :py:class:`~xarray.DataArray`
        The created pseudo observations based on the given state and this
        observation operator. The last two dimensions are the same
        dimensions as the ``observations`` :py:class:`~xarray.DataArray`
        in set observation subset.
    """
    pseudo_obs = state.sel(var_name='x')
    pseudo_obs = pseudo_obs.rename(grid='obs_grid_1')
    pseudo_obs['time'] = obs_ds.time.values
    pseudo_obs['obs_grid_1'] = obs_ds.obs_grid_1.values
    return pseudo_obs


def dummy_update_state(self, state, observations, pseudo_state, analysis_time):
    """
    This dummy update state can be used to patch
    :py:meth:`~pytassim.assimilation.base.BaseAssimilation.update_state` for
    testing purpose. This dummy update state slices state by nearest
    ``analysis_time``.

    Parameters
    ----------
    self : obj
        This method is patched to this object.
    state : :py:class:`xarray.DataArray`
        This state is used to generate an observation-equivalent. It is
        further updated by this assimilation algorithm and given
        ``observation``. This :py:class:`~xarray.DataArray` should have
        four coordinates, which are specified in
        :py:class:`pytassim.state.ModelState`.
    observations : :py:class:`xarray.Dataset` or \
    iterable(:py:class:`xarray.Dataset`)
        These observations are no used in this dummy method.
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
    analysis = state.sel(time=analysis_time, method='nearest')
    analysis = analysis.expand_dims('time', axis=1)
    return analysis


def dummy_model(state):
    """
    This model is a dummy model which can be used for testing purpose. This
    dummy model only returns given state without type testing such that it can
    be used for all types of inputs.

    Parameters
    ----------
    state : any
        This `model_state` is returned by this dummy_model.

    Returns
    -------
    derivative : any
        The returned state, same as given ``state``.
    """
    derivative = state
    return derivative


class DummyLocalization(object):
    """
    This localization selects only grid points where `grid_ind` and `obs_grid`
    are the same.
    """
    @staticmethod
    def localize_obs(grid_ind, obs_grid):
        distance = np.abs(obs_grid-grid_ind)
        obs_weights = np.clip(1 - distance / 10, a_min=0, a_max=None)[:, 0]
        use_obs = obs_weights > 0
        return use_obs, obs_weights


def dummy_distance(a, b):
    """
    This calculates a dummy distance as :math:`\\text{abs}(a-b)`.

    Returns
    -------
    distance : any
    """
    distance = np.abs(a-b)
    return distance


class DummyNeuralModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8, bias=False)

    def forward(self, in_tensor):
        return self.linear(in_tensor)

    @staticmethod
    def assimilate(state, *args, **kwargs):
        return state
