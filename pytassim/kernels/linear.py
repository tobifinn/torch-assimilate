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

# Internal modules
from .base_kernels import BaseKernel
from . import utils as kernel_utils


logger = logging.getLogger(__name__)


__all__ = ['LinearKernel']


class LinearKernel(BaseKernel):
    """
    This linear kernel specifies a dot product between given matrices. The
    linear kernel can be used to construct a Bayesian linear regression. The
    kernelized ensemble transform Kalman filter (
    :py:class:`pytassim.assimilation.filter.ketkf.KETKFUncorr`) with this
    kernel reconstructs the standard ensemble transform Kalman filter (
    :py:class:`pytassim.assimilation.filter.etkf.ETKFUncorr`). This kernel is
    not stationary and degenerated, meaning that it is not translation-invariant
    and might lead to inconsistent results in non-linear regimes.


    .. math::

       K(x_i, x_j) = {(x_i)}^{T} x_j

    """
    def __str__(self) -> str:
        return 'LinearKernel'

    def __repr__(self) -> str:
        return 'Linear'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return kernel_utils.dot_product(x, y)
