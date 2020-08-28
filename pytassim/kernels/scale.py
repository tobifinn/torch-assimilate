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


logger = logging.getLogger(__name__)


__all__ = ['ScaleKernel']


class ScaleKernel(BaseKernel):
    """
    This scale kernel specify a constant value :math:`c`, which can be used to
    specify the output variance of the kernel or a constant shift,


    .. math::

       K(x_i, x_j) = c


    Parameters
    ----------
    scaling : torch.Tensor, optional
        This scaling factor specifies the constant value and is deactivated
        in its default value of 0.

    """
    def __init__(self, scaling: torch.Tensor = torch.tensor(0.)):
        super().__init__()
        self.scaling = scaling

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        const_tensor = self.scaling * torch.ones((x.shape[0], y.shape[0]))
        return const_tensor
