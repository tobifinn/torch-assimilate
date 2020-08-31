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
import torch.nn

# Internal modules
from .base_kernels import BaseKernel
from .utils import dot_product


logger = logging.getLogger(__name__)


__all__ = ['ModuleKernel']


class ModuleKernel(BaseKernel):
    """
    This module kernel can be used to specify an explicit feature
    transformation process. This kernel takes a :py:class:`torch.nn.Module`
    :math:`\\phi`, which is then applied to the given data. The result is
    combined with a linear kernel as follows,


    .. math::

       K(x_i, x_j) = {(\\phi(x_i))}^{T} \\phi(x_y).

    This kernel can be used to implement random fourier features
    :cite:`rahimi_random_2008`, instances of the Nystrom method
    :cite:`drineas_nystrom_2005` or feature extraction with neural networks.

    Parameters
    ----------
    transform_module : :py:class:`torch.nn.Module`
        This transform module is used to transform given data into a new
        feature space.

    """
    def __init__(self, transform_module: torch.nn.Module):
        super().__init__()
        self.add_module('transform', transform_module)

    def __str__(self) -> str:
        return 'ModuleKernel({0})'.format(str(self.transform))

    def __repr__(self) -> str:
        return 'ModuleKernel'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_net = self.transform(x)
        y_net = self.transform(y)
        k_mat = dot_product(x_net, y_net)
        return k_mat
