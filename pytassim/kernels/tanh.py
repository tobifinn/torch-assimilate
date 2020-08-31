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
from .utils import dot_product


logger = logging.getLogger(__name__)


__all__ = ['TanhKernel']


class TanhKernel(BaseKernel):
    """
    This tangens hyperbolicus kernel represents a multilayer perceptron
    kernel and often also called sigmoid kernel :cite:`lin_study_2003`. The
    dot product of the data is multiplied a coefficient :math:`\\alpha` and
    an additional intercept constant :math:`c` is added, before the product
    is transformed by a tanh function, which is a scaled sigmoid,


    .. math::

       K(x_i, x_j) = \\tanh(\\alpha {(x_i)}^{T} x_j + c)


    Parameters
    ----------
    coeff : torch.Tensor, optional
        This coefficient is used to scale the dot product output and is set
        to 1, resulting in a not scaled dot product.
    const : torch.Tensor, optional
        This constant intercept value is used to shift the output of the dot
        product, before the tanh function is applied. The default value of 0
        deactivates shifting.

    """
    def __init__(self, coeff: torch.Tensor = torch.tensor(1.),
                 const: torch.Tensor = torch.tensor(0.)):
        super().__init__()
        self.coeff = coeff
        self.const = const

    def __str__(self) -> str:
        return 'TanhKernel({0}, {1})'.format(
            str(self.coeff), str(self.const)
        )

    def __repr__(self) -> str:
        return 'Tanh({0}, {1})'.format(repr(self.coeff), repr(self.const))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = dot_product(x, y)
        logit = self.coeff * xy + self.const
        kernel_mat = torch.tanh(logit)
        return kernel_mat
