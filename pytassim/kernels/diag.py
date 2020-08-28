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


class DiagKernel(BaseKernel):
    """
    The diagonal kernel
    """
    def __init__(self, scaling: torch.Tensor = torch.tensor(0.)):
        super().__init__()
        self.scaling = scaling

    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        if x.shape[-2] != y.shape[-2]:
            k_mat = torch.zeros(x.shape[:-1] + (y.shape[-2],))
        else:
            k_mat = torch.ones(x.shape[:-1])
            k_mat = torch.diag_embed(k_mat) * self.scaling
        k_mat = k_mat.to(x)
        return k_mat
