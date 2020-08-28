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
from typing import Any, Tuple

# External modules
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


class BaseLocalization(object):
    """
    This base localization should be used if a localization algorithm is
    implemented.
    """
    @abc.abstractmethod
    def localize_cov(self):
        """
        This method creates a localized covariance based on given parameters.
        """
        pass

    @abc.abstractmethod
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
        pass
