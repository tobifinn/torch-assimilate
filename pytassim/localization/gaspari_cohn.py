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
import warnings
from typing import Tuple, Union, Callable, Any

# External modules
import numpy as np

# Internal modules
from .localization import BaseLocalization


logger = logging.getLogger(__name__)


class GaspariCohn(BaseLocalization):
    """
    This localization can  be used to constrain observations. It is based on
    Gaspari-Cohn correlation function :cite:`gaspari_construction_1999`. This
    correlation function corresponds to a form factor of :math:`\\frac{1}{2}`,
    in :cite:`gaspari_construction_1999` :math:`C_0(z, \\frac{1}{2}, c)`.

    Parameters
    ----------
    length_scale : float or tuple(float)
        This length scale is :math:`c` in :cite:`gaspari_construction_1999`.
        This length scale determines
        the truncation to zero and the decay of the covariance function. The
        Gaspari-Cohn function is truncated to zero by 2 * length_scale.
    dist_func : func
        This distance function is used to determine the distance between states.
        This functions takes two different grid lists and estimates a distance
        between these two grids.
    """
    def __init__(
            self,
            length_scale: Union[float, Tuple[float]],
            dist_func: Callable,
            epsilon: float = 1E-5
    ):
        self.radius = np.atleast_1d(length_scale)
        self.dist_func = dist_func
        self.epsilon = epsilon
        self._thres = [2, 1]

    def __str__(self) -> str:
        return 'GaspariCohn(l={0})'.format(str(self.radius))

    def __repr__(self) -> str:
        return 'GaspariCohn'

    @staticmethod
    def _f1(dist: np.ndarray) -> np.ndarray:
        f1 = - 0.25 * dist ** 5
        f1 += 0.5 * dist ** 4
        f1 += 0.625 * dist ** 3
        f1 -= 5 / 3 * dist ** 2
        f1 += 1
        return f1

    @staticmethod
    def _f2(dist: np.ndarray) -> np.ndarray:
        f2 = 1 / 12 * dist ** 5
        f2 -= 0.5 * dist ** 4
        f2 += 0.625 * dist ** 3
        f2 += 5 / 3 * dist ** 2
        f2 -= 5 * dist
        f2 += 4
        f2 -= 2 / 3 / dist
        return f2

    def localize_obs(
            self,
            grid_ind: Any,
            obs_grid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method creates weights for observations based on given grid index
        and observation grid.

        Parameters
        ----------
        grid_ind : any
            This parameter indicates the current grid index.
        obs_grid : :py:class:`np.ndarray`
            This observation grid is used to estimate a spatial distance to
            given grid index.

        Returns
        -------
        use_obs : :py:class:`np.ndarray`, dtype=bool
            This is a boolean array indicating if the `i`-th observation should
            be used. This array is calculated based on estimated observation
            weights.
        obs_weights : :py:class:`np.ndarray`, dtype=float
            The estimated observation weights. These weights can be used to
            weight observations.
        """
        weights = np.ones((obs_grid.shape[0]), dtype=float)
        dist = np.atleast_2d(self.dist_func(grid_ind, obs_grid))
        for i, d in enumerate(dist):
            dist_radius = d / self.radius[i]
            conds = [dist_radius < thres for thres in self._thres]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmp_weights = np.zeros((obs_grid.shape[0]), dtype=float)
                tmp_weights[conds[0]] = self._f2(dist_radius[conds[0]])
                tmp_weights[conds[1]] = self._f1(dist_radius[conds[1]])
            weights *= tmp_weights
        use_obs = weights > self.epsilon
        return use_obs, weights


class GaspariCohnInf(BaseLocalization):
    """
    This localization can  be used to constrain observations. It is based on
    Gaspari-Cohn correlation function :cite:`gaspari_construction_1999`. This
    correlation function corresponds to a form factor of infinity, in
    :cite:`gaspari_construction_1999` :math:`C_0(z, \\infty, c)`.

    Parameters
    ----------
    length_scale : float or tuple(float)
        This length scale is :math:`c` in :cite:`gaspari_construction_1999`.
        This length scale determines
        the truncation to zero and the decay of the covariance function. The
        Gaspari-Cohn function is truncated to zero by 2 * length_scale.
    dist_func : func
        This distance function is used to determine the distance between states.
        This functions takes two different grid lists and estimates a distance
        between these two grids.
    """
    def __init__(
            self,
            length_scale: Union[float, Tuple[float]],
            dist_func: Callable,
            epsilon: float = 1E-5
    ):
        self.radius = length_scale
        self.dist_func = dist_func
        self.epsilon = epsilon
        self._thres = [2, 1.5, 1, 0.5]

    def __str__(self) -> str:
        return 'GaspariCohnInf(l={0})'.format(str(self.radius))

    def __repr__(self) -> str:
        return 'GaspariCohnInf'

    @staticmethod
    def _f1(dist: np.ndarray) -> np.ndarray:
        f1 = -28 * dist ** 5 / 33
        f1 += 8 * dist ** 4 / 11
        f1 += 20 * dist ** 3 / 11
        f1 -= 80 * dist ** 2 / 33
        f1 += 1
        return f1

    @staticmethod
    def _f2(dist: np.ndarray) -> np.ndarray:
        f2 = 20 * dist ** 5 / 33
        f2 -= 16 * dist ** 4 / 11
        f2 += 100 * dist ** 2 / 33
        f2 -= 45 * dist / 11
        f2 += 51 / 22
        f2 -= 7 / (44 * dist)
        return f2

    @staticmethod
    def _f3(dist: np.ndarray) -> np.ndarray:
        f3 = -4 * dist ** 5 / 11
        f3 += 16 * dist ** 4 / 11
        f3 -= 10 * dist ** 3 / 11
        f3 -= 100 * dist ** 2 / 33
        f3 += 5 * dist
        f3 -= 61 / 22
        f3 += 115 / (132 * dist)
        return f3

    @staticmethod
    def _f4(dist: np.ndarray) -> np.ndarray:
        f4 = 4 * dist ** 5 / 33
        f4 -= 8 * dist ** 4 / 11
        f4 += 10 * dist ** 3 / 11
        f4 += 80 * dist ** 2 / 33
        f4 -= 80 * dist / 11
        f4 += 64 / 11
        f4 -= 32 / (33 * dist)
        return f4

    def localize_obs(
            self,
            grid_ind: Any,
            obs_grid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method creates weights for observations based on given grid index
        and observation grid.

        Parameters
        ----------
        grid_ind : any
            This parameter indicates the current grid index.
        obs_grid : :py:class:`np.ndarray`
            This observation grid is used to estimate a spatial distance to
            given grid index.

        Returns
        -------
        use_obs : :py:class:`np.ndarray`, dtype=bool
            This is a boolean array indicating if the `i`-th observation should
            be used. This array is calculated based on estimated observation
            weights.
        obs_weights : :py:class:`np.ndarray`, dtype=float
            The estimated observation weights. These weights can be used to
            weight observations.
        """
        weights = np.zeros((obs_grid.shape[-1]), dtype=float)
        dist = self.dist_func(grid_ind, obs_grid)
        dist_radius = dist / self.radius
        conds = [dist_radius < thres for thres in self._thres]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weights[conds[0]] = self._f4(dist_radius[conds[0]])
            weights[conds[1]] = self._f3(dist_radius[conds[1]])
            weights[conds[2]] = self._f2(dist_radius[conds[2]])
        weights[conds[3]] = self._f1(dist_radius[conds[3]])
        use_obs = weights > self.epsilon
        return use_obs, weights
