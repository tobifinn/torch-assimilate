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


__all__ = ['DiagKernel']


class DiagKernel(BaseKernel):
    """
    The diagonal kernel specifies a diagonal matrix, if the number of samples
    for the given `x` and `y` is the same. If the numbers of samples differ,
    then a zero matrix is returned. This type of Kernel can be used to
    specify white noise for the observational uncertainty.

    Parameters
    ----------
    scaling : torch.Tensor, optional
        The diagonal matrix is multiplied with this scaling factor, which can
        be used to specify a noise level (default=0.).
    """
    def __init__(self, scaling: torch.Tensor = torch.tensor(0.)):
        super().__init__()
        self.scaling = scaling

    def __str__(self) -> str:
        return 'DiagKernel({0})'.format(str(self.scaling))

    def __repr__(self) -> str:
        return 'Diag({0})'.format(repr(self.scaling))

    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        if x.shape[-2] != y.shape[-2]:
            k_mat = torch.zeros(x.shape[:-1] + (y.shape[-2],))
        else:
            k_mat = torch.ones(x.shape[:-1])
            k_mat = torch.diag_embed(k_mat) * self.scaling
        k_mat = k_mat.to(x)
        return k_mat
