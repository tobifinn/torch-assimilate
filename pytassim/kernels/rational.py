#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 03.08.20
#
# Created for 20_kernel_etkf
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}
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
import torch
import torch.nn

# Internal modules
from .base_kernels import BaseKernel
from .utils import euclidean_dist


logger = logging.getLogger(__name__)


__all__ = ['RationalKernel']


class RationalKernel(BaseKernel):
    """
    The rational kernel is similar to apply radial basis function kernels
    with different lengthscales. This kernel has a basis lengthscale
    :math:`l` and a weighting factor :math:`a`, which weights small-scale to
    large-scale variations (:cite:`duvenaud_automatic_2014`),

    .. math::

       K(x_i, x_j) = (1 + \\frac{{(x_i-x_j)}^2}{2\\,a\\,l^2})^{-a}.


    Parameters
    ----------
    lengthscale : torch.Tensor, optional
        This lengthscale specifies the scale of variations. The
        default value of 1 assumes that the input is already normalized.
    weighting : torch.Tensor, optional
        This weighting factor specifies the relative weighting between
        small-scale to large-scale variations (default=1.)

    """
    def __init__(self, lengthscale: torch.Tensor = torch.tensor(1.),
                 weighting: torch.Tensor = torch.tensor(1.)):
        super().__init__()
        self.lengthscale = lengthscale
        self.weighting = weighting

    def __str__(self) -> str:
        return 'RationalKernel({0}, {1})'.format(
            str(self.lengthscale), str(self.weighting)
        )

    def __repr__(self) -> str:
        return 'Rational({0}, {1})'.format(repr(self.lengthscale),
                                           repr(self.weighting))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_scaled = x.div(self.lengthscale)
        y_scaled = y.div(self.lengthscale)
        euc_dist = euclidean_dist(x_scaled, y_scaled)
        norm_factor = 2 * self.weighting
        factor = 1 + euc_dist / norm_factor
        k_mat = factor.pow(-self.weighting)
        return k_mat
