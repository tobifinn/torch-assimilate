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
from .utils import dot_product


logger = logging.getLogger(__name__)


__all__ = ['PolyKernel']


class PolyKernel(BaseKernel):
    """
    This polynomial kernel specifies a :math:`p` polynomial feature space,
    constructed based on given data with constant value :math:`c`,


    .. math::

       K(x_i, x_j) = ({(x_i)}^{T} x_j + c)^{p}


    Parameters
    ----------
    degree : torch.Tensor, optional
        This specifies the degree of the polynomial. The default of 2
        corresponds to a quadratic kernel.
    const : torch.Tensor, optional
        This constant value is used to shift the center of the kernel. The
        default of 1 activates shifting of the kernel center.

    """
    def __init__(self, degree: torch.Tensor = torch.tensor(2.),
                 const: torch.Tensor = torch.tensor(1.)):
        super().__init__()
        self.degree = degree
        self.const = const

    def __str__(self) -> str:
        return 'PolynomialKernel({0}, {1})'.format(
            str(self.degree), str(self.const)
        )

    def __repr__(self) -> str:
        return 'Polynomial({0}, {1})'.format(repr(self.degree),
                                             repr(self.const))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = dot_product(x, y)
        k_mat = (xy + self.const).pow(self.degree)
        return k_mat
