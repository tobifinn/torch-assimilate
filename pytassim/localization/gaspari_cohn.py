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

# Internal modules
from .localization import BaseLocalization


logger = logging.getLogger(__name__)


class GaspariCohn(BaseLocalization):
    """
    This localization can  be used to constrain observations. It is based on
    Gaspari-Cohn correlation function [GC99]_. This correlation function is
    similar to a gaussian with truncation.

    References
    ----------
    .. [GC99] Gaspari, G., & Cohn, S. E. (1999).
              Construction of correlation functions in two and three dimensions.
              Quarterly Journal of the Royal Meteorological Society, 125(554),
              723–757.

    Parameters
    ----------
    length_scale : float or tuple(float)
        This length scale is :math:`c` in [GC99]_. This length scale determines
        the truncation to zero and the decay of the covariance function. The
        Gaspari-Cohn function is truncated to zero by 2 * length_scale.
    dist_func : func
        This distance function is used to determine the distance between states.
        This functions takes two different grid lists and estimates a distance
        between these two grids.
    """
    def __init__(self, length_scale, dist_func):
        self.radius = length_scale
        self.dist_func = dist_func

    @staticmethod
    def _f1(dist):
        f1 = -28 * dist ** 5 / 33
        f1 += 8 * dist ** 4 / 11
        f1 += 20 * dist ** 3 / 11
        f1 -= 80 * dist ** 2 / 33
        f1 += 1
        return f1

    @staticmethod
    def _f2(dist):
        f2 = 20 * dist ** 5 / 33
        f2 -= 16 * dist ** 4 / 11
        f2 += 100 * dist ** 2 / 33
        f2 -= 45 * dist / 11
        f2 += 51 / 22
        f2 -= 7 / (44 * dist)
        return f2

    @staticmethod
    def _f3(dist):
        f3 = -4 * dist ** 5 / 11
        f3 += 16 * dist ** 4 / 11
        f3 -= 10 * dist ** 3 / 11
        f3 -= 100 * dist ** 2 / 33
        f3 += 5 * dist
        f3 -= 61 / 22
        f3 += 115 / (132 * dist)
        return f3

    @staticmethod
    def _f4(dist):
        f4 = 4 * dist ** 5 / 33
        f4 -= 8 * dist ** 4 / 11
        f4 += 10 * dist ** 3 / 11
        f4 += 80 * dist ** 2 / 33
        f4 -= 80 * dist / 11
        f4 += 64 / 11
        f4 -= 32 / (33 * dist)
        return f4

    def localize_obs(self, grid_ind, obs_grid):
        pass
