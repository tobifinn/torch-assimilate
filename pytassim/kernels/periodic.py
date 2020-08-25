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


class PeriodicKernel(BaseKernel):
    def __init__(self, period=1., lengthscale=1.):
        super().__init__()
        self.period = period
        self.lengthscale = lengthscale

    def forward(self, x, y):
        dist_mat = distance_matrix(x, y, 1.) * np.pi / self.period
        factor = -2 * torch.sin(-dist_mat).pow(2) / (self.lengthscale ** 2)
        return torch.exp(factor)
