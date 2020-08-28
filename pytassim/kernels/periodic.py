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

import numpy as np

# Internal modules
from .base_kernels import BaseKernel
from .utils import distance_matrix


logger = logging.getLogger(__name__)


__all__ = ['PeriodicKernel']


class PeriodicKernel(BaseKernel):
    """
    The periodic kernel specifies a periodic process with a given periodicity
    :math:`p` and a lengthscale :math:`l` :cite:`duvenaud_automatic_2014`,


    .. math::

       K(x_i, x_j) = \\exp(-\\frac{2 \\sin^2(\\pi\\mid\\mid x_i - x_j
       \\mid\\mid_1/p)}{l^2})


    Parameters
    ----------
    period : torch.Tensor, optional
        This specifies the periodicity of the kernel (default=1)
    lengthscale : torch.Tensor, optional
        This specifies the length scale, similar to the
        :py:class:`pytassim.kernel.rbf.GaussKernel` (default=1)

    """
    def __init__(self, period: torch.Tensor = torch.tensor(1.),
                 lengthscale: torch.Tensor = torch.tensor(1.)):
        super().__init__()
        self.period = period
        self.lengthscale = lengthscale

    def __str__(self) -> str:
        return 'PeriodicKernel({0}, {1})'.format(
            str(self.period), str(self.lengthscale)
        )

    def __repr__(self) -> str:
        return 'Periodic({0}, {1})'.format(repr(self.period),
                                           repr(self.lengthscale))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist_mat = distance_matrix(x, y, 1.) * np.pi / self.period
        factor = -2 * torch.sin(-dist_mat).pow(2) / (self.lengthscale ** 2)
        return torch.exp(factor)
