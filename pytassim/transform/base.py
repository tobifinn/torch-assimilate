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


class BaseTransformer(object):
    """
    The transformer can be used to transform data before (pre) or after (post)
    assimilation. This can be either some type of covariance inflation,
    normalization or something completely different.
    """
    @abc.abstractmethod
    def pre(self, background, observations, first_guess):
        """
        This transform method is called before the assimilation and can change
        input ``first_guess`` and ``observations``.
        Parameters
        ----------
        background : :py:class:`xarray.DataArray`
            This background array is manipulated by this method and should be
            a valid state array.
        observations : iterable(:py:class:`xarray.Dataset`
            These observations are manipulated by this method and should be
            valid observation datasets with observations and covariance as
            array.
        first_guess : :py:class:`xarray.DataArray`
            This first guess array is manipulated by this method and should be
            a valid state array.

        Returns
        -------
        background : :py:class:`xarray.DataArray`
            This background array was manipulated by this method and should be
            a valid state array.
        observations : iterable(:py:class:`xarray.Dataset`
            These observations were manipulated by this method and are
            valid observation datasets with observations and covariance as
            array.
        first_guess : :py:class:`xarray.DataArray`
            This first guess field was manipulated by this method and is a valid
            state array.
        """
        pass

    @abc.abstractmethod
    def post(self, analysis, background, observations, first_guess):
        """
        This transform method is called after the assimilation and can change
        input ``analysis``. Additional ``first_guess`` and ``observations`` can
        be used to manipulate the analysis.

        Parameters
        ----------
        analysis : :py:class:`xarray.DataArray`
            This analysis array is manipulated by this method and should be
            a valid state array.
        background : :py:class:`xarray.DataArray`
            This background array can be used to manipulate given analysis and
            should be a valid state array.
        observations : iterable(:py:class:`xarray.Dataset`
            These observations can be used to manipulate the analysis and should
            be valid observation datasets with observations and covariance as
            array.
        first_guess : :py:class:`xarray.DataArray`
            This first guess array can be used to manipulate given analysis and
            should be a valid state array.

        Returns
        -------
        analysis : :py:class:`xarray.DataArray`
            This analysis array was manipulated by this method and is a valid
            state array.
        """
        pass
