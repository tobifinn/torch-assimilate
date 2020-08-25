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
import abc

# External modules
import torch.nn

# Internal modules


logger = logging.getLogger(__name__)


class BaseKernel(torch.nn.Module):
    def __add__(self, other):
        return AdditiveKernel(self, other)

    def __mul__(self, other):
        return MultiplicativeKernel(self, other)

    def __pow__(self, other):
        return PowerKernel(self, other)

    @abc.abstractmethod
    def forward(self, x, y):
        pass


class CompKernel(BaseKernel):
    def __init__(self, kernel_1, kernel_2):
        super().__init__()
        self.add_module('kernel_1', kernel_1)
        self.add_module('kernel_2', kernel_2)


class AdditiveKernel(CompKernel):
    def forward(self, x, y):
        return self.kernel_1(x, y) + self.kernel_2(x, y)


class MultiplicativeKernel(CompKernel):
    def forward(self, x, y):
        return self.kernel_1(x, y) * self.kernel_2(x, y)


class PowerKernel(CompKernel):
    def forward(self, x, y):
        return self.kernel_1(x, y).pow(self.kernel_2(x, y))
