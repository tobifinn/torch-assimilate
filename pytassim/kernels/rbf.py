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


__all__ = ['GaussKernel', 'RBFKernel']


class GaussKernel(BaseKernel):
    """
    The Gaussian kernel is a type of radial basis function
    kernel :py:class:`pytassim.kernels.rbf.RBFKernel` and is implemented
    with a given lengthscale :math:`l`,

    .. math::

       K(x_i, x_j) = \\exp(-\\frac{{(x_i-x_j)}^2}{2\\,l^2}).


    Parameters
    ----------
    lengthscale : torch.Tensor, optional
        This lengthscale is used to estimate the Gaussian kernel. The
        default value of 1 assumes that the input is already normalized.
    """

    def __init__(self, lengthscale: torch.Tensor = torch.tensor(1.)):
        super().__init__()
        self.lengthscale = lengthscale

    def __str__(self):
        return "GaussKernel(l={0})".format(self.lengthscale)

    def __repr__(self):
        return "GaussKernel"

    def _get_lengthscale(self) -> torch.Tensor:
        return self.lengthscale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_scaled = x.div(self._get_lengthscale())
        y_scaled = y.div(self._get_lengthscale())
        euc_dist = euclidean_dist(x_scaled, y_scaled)
        factor = euc_dist / 2.
        k_mat = torch.exp(-factor)
        return k_mat


class RBFKernel(GaussKernel):
    """
    This radial basis function kernel is a universal kernel and is
    dependent on the chosen `gamma` :math:`\\gamma` factor,

    .. math::

       K(x_i, x_j) = \\exp(-\\gamma\\,{(x_i-x_j)}^2).

    Parameters
    ----------
    gamma : torch.Tensor, optional
        This gamma value if  is used to estimate the Gaussian kernel. The
        default value of 0.5 assumes that the input is already normalized.

    """
    def __init__(self, gamma: torch.Tensor = torch.tensor(0.5)):
        super().__init__()
        self.gamma = gamma

    def __str__(self):
        return "RBFKernel(\u03B3={0})".format(self.gamma)

    def __repr__(self):
        return "RBFKernel"

    def _get_lengthscale(self) -> torch.Tensor:
        return (0.5 / self.gamma) ** 0.5
