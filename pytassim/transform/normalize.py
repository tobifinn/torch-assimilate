#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12/12/18
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
import types
import copy

# External modules

# Internal modules
from .base import BaseTransformer


logger = logging.getLogger(__name__)


class Normalizer(BaseTransformer):
    """
    This Normalizer normalize the first guess and observations before
    assimilation and reverts the normalization of the analysis after the
    assimilation. These methods can be used to generate normalized in- and
    outputs for machine learning algorithms. To normalize the mean value
    :math:`\\mu`` is subtracted from given field :math:`x`, which is then
    divided by given standard deviation :math:`\\sigma`,

    .. math::

       x_{\\text{norm}} = \\frac{x - \\mu}{\\sigma}.

    For post-processing this formula is reverted.



    Parameters
    ----------
    ens_stat : iterable(any)
        These ensemble statistics are used to normalize the background and to
        revert the normalization of the analysis. The first item of this
        iterable is used as mean value, while the second item is used as
        standard deviation.
    obs_stat : iterable(any)
        These observation statistics are used to normalize the observations
        before assimilation. This iterable should have at least the same length
        as given observations and is used in the same order as given
        observations. Every item should be an iterable, which is used for mean
        (first item) and standard deviation (second item).
    fg_stat : iterable(any)
        These first guess statistics are used to normalize the first guess
        array. The first item of this iterable is used as mean value, while the
        second item is used as standard deviation.
    """
    def __init__(self, ens_stat, obs_stat, fg_stat):
        self.ens_stat = ens_stat
        self.obs_stat = obs_stat
        self.fg_stat = fg_stat

    def pre(self, background, observations, first_guess):
        """
        This method normalizes given `background`, `observations` and
        `first_guess`by their set statistics with

        .. math::

           x_{\\text{norm}} = \\frac{x - \\mu}{\\sigma}.


        Parameters
        ----------
        background : :py:class:`xarray.DataArray`
            This array is normalized by set ensemble statistics. This first
            guess should be a valid state.
        observations : iterable(:py:class:`xarray.Dataset`)
            These observations are normalized by set observation statistics. To
            normalize these observations, the `observation` values are centered
            and scaled. These observations must be in the same order as set
            observation statistics. These observations should be valid
            observations with `observation` and `covariance` as
            :py:class:`xarray.DataArray`.
        first_guess : :py:class:`xarray.DataArray`
            This array is normalized by set ensemble statistics. This first
            guess should be a valid state.

        Returns
        -------
        background : :py:class:`xarray.DataArray`
            The centered and scaled background field.
        observations : iterable(:py:class:`xarray.Dataset`)
            The normalized observations with centered and scaled observation
            values.
        first_guess : :py:class:`xarray.DataArray`
            The centered and scaled first guess field.
        """
        background = (background - self.ens_stat[0]) / self.ens_stat[1]
        first_guess = (first_guess - self.fg_stat[0]) / self.fg_stat[1]
        obs_list = []
        for k, obs in enumerate(observations):
            tmp_obs = obs.copy(deep=True)
            tmp_obs['observations'] -= self.obs_stat[k][0]
            tmp_obs['observations'] /= self.obs_stat[k][1]
            tmp_obs.obs.operator = obs.obs.operator
            obs_list.append(tmp_obs)
        return background, obs_list, first_guess

    def post(self, analysis, background, observations, first_guess):
        """
        This method reverts the normalization of the analysis based on set
        ensemble statistics.

        Parameters
        ----------
        analysis : :py:class:`xarray.DataArray`
            The normalization of this analysis field is reverted. This should be
            a valid state.
        background : :py:class:`xarray.DataArray`
            This background is not used in this normalization method.
        observations : iterable(:py:class:`xarray.Dataset`)
            These observations are not used in this normalization method.
        first_guess : :py:class:`xarray.DataArray`
            This first guess is not used in this normalization method.

        Returns
        -------
        analysis : :py:class:`xarray.DataArray`
            This analysis was amplified by set standard deviation and shifted by
            set mean. This is a valid state.
        """
        analysis = analysis * self.ens_stat[1] + self.ens_stat[0]
        return analysis
