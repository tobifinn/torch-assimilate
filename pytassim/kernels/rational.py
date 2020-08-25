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


class RationalKernel(BaseKernel):
    def __init__(self, lengthscale=1., weighting=1.):
        super().__init__()
        self.lengthscale = lengthscale
        self.weighting = weighting

    def forward(self, x, y):
        euc_dist = euclidean_dist(x, y)
        norm_factor = 2 * self.weighting * self.lengthscale
        factor = 1 + euc_dist / norm_factor
        k_mat = factor.pow(-self.weighting)
        return k_mat
