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
from .utils import distance_matrix


logger = logging.getLogger(__name__)


__all__ = ['OrnsteinUhlenbeckKernel']


class OrnsteinUhlenbeckKernel(BaseKernel):
    """
    This kernel specifies an Ornstein-Uhlenbeck random process, representing
    a continuous random walk. The specified lengthscale :math:`l` specifies
    the step length,

    .. math::

       K(x_i, x_j) = \\exp(-\\frac{\\mid\\mid x_i - x_j \\mid\\mid_1}{l})


    Parameters
    ----------
    lengthscale : torch.Tensor, optional
        This lengthscale is used to estimate this kernel. The
        default value of 1 assumes that the input is already normalized.

    """
    def __init__(self, lengthscale: torch.Tensor = torch.tensor(1.)):
        super().__init__()
        self.lengthscale = lengthscale

    def __str__(self) -> str:
        return 'OrnsteinUhlenbeckKernel({0})'.format(str(self.lengthscale))

    def __repr__(self) -> str:
        return 'OrnUhlKernel({0})'.format(repr(self.lengthscale))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        abs_dist = distance_matrix(x, y, norm=1)
        factor = -abs_dist / self.lengthscale
        k_mat = torch.exp(factor)
        return k_mat
