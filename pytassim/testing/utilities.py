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

# Internal modules


logger = logging.getLogger(__name__)


def dummy_obs_operator():
    pass


def dummy_update_state(self, state, observations, analysis_time):
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
