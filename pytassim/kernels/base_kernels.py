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
    """
    This kernel is the base kernel for all other kernel objects and used to
    add mathematical operations like `__add__`, `__mul__` and `__power__`.
    New kernels should be a child of this BaseKernel and have to overwrite
    the `forward(self, x, y)` method.
    """
    def __add__(self, other: torch.nn.Module) -> torch.nn.Module:
        return AdditiveKernel(self, other)

    def __mul__(self, other: torch.nn.Module) -> torch.nn.Module:
        return MultiplicativeKernel(self, other)

    def __pow__(self, other: torch.nn.Module) -> torch.nn.Module:
        return PowerKernel(self, other)

    @abc.abstractmethod
    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        pass


class CompKernel(BaseKernel):
    """
    This composition kernel is used as base kernel for all types of
    compositions, like addition or multiplication. This composition adds two
    given kernels as :py:class:`torch.nn.Module`, which might be then used to
    compose a new kernel from the given kernels.

    Parameters
    ----------
    kernel_1 : child of :py:class:`pytassim.kernels.base_kernels.BaseKernel`
        This is the first kernel, which is used to compose a given new kernel.
    kernel_2 : child of :py:class:`pytassim.kernels.base_kernels.BaseKernel`
        This is the second kernel, which is used to compose a given new kernel.

    """
    def __init__(self, kernel_1: BaseKernel, kernel_2: BaseKernel):
        super().__init__()
        self.add_module('kernel_1', kernel_1)
        self.add_module('kernel_2', kernel_2)


class AdditiveKernel(CompKernel):
    """
    The additive kernel adds the second given kernel to the first given kernel.

    .. math::

       K(x_i, x_j) = K_1(x_i, x_j) + K_2(x_i, x_j)


    Parameters
    ----------
    kernel_1 : child of :py:class:`pytassim.kernels.base_kernels.BaseKernel`
        This is the first kernel, which is used to compose a given new kernel.
    kernel_2 : child of :py:class:`pytassim.kernels.base_kernels.BaseKernel`
        This is the second kernel, which is used to compose a given new kernel.

    """
    def __str__(self) -> str:
        return "{0:s}+{1:s}".format(str(self.kernel_1), str(self.kernel_2))

    def __repr__(self) -> str:
        return "{0:s}+{1:s}".format(repr(self.kernel_1), repr(self.kernel_2))

    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        return self.kernel_1(x, y) + self.kernel_2(x, y)


class MultiplicativeKernel(CompKernel):
    """
    The multiplicative kernel multiplies the second given kernel with the first
    given kernel.

    .. math::

       K(x_i, x_j) = K_1(x_i, x_j) * K_2(x_i, x_j)


    Parameters
    ----------
    kernel_1 : child of :py:class:`pytassim.kernels.base_kernels.BaseKernel`
        This is the first kernel, which is used to compose a given new kernel.
    kernel_2 : child of :py:class:`pytassim.kernels.base_kernels.BaseKernel`
        This is the second kernel, which is used to compose a given new kernel.

    """
    def __str__(self) -> str:
        return "{0:s}*{1:s}".format(str(self.kernel_1), str(self.kernel_2))

    def __repr__(self) -> str:
        return "{0:s}*{1:s}".format(repr(self.kernel_1), repr(self.kernel_2))

    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        return self.kernel_1(x, y) * self.kernel_2(x, y)


class PowerKernel(CompKernel):
    """
    The power kernel takes the first given kernel to the power of the second
    given kernel.

    .. math::

       K(x_i, x_j) = {K_1(x_i, x_j)}^{K_2(x_i, x_j)}


    Parameters
    ----------
    kernel_1 : child of :py:class:`pytassim.kernels.base_kernels.BaseKernel`
        This is the first kernel, which is used to compose a given new kernel.
    kernel_2 : child of :py:class:`pytassim.kernels.base_kernels.BaseKernel`
        This is the second kernel, which is used to compose a given new kernel.

    """
    def __str__(self) -> str:
        return "{0:s}^{1:s}".format(str(self.kernel_1), str(self.kernel_2))

    def __repr__(self) -> str:
        return "{0:s}^{1:s}".format(repr(self.kernel_1), repr(self.kernel_2))


    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        return self.kernel_1(x, y).pow(self.kernel_2(x, y))
