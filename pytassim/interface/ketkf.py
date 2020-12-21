#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 21.12.20
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union, Iterable

# External modules
import torch
import torch.nn

# Internal modules
from .etkf import ETKF
from ..kernels.base_kernels import BaseKernel
from ..kernels.linear import LinearKernel
from ..core.ketkf import KETKFModule
from ..transform.base import BaseTransformer


logger = logging.getLogger(__name__)


class KETKF(ETKF):
    def __init__(
            self,
            kernel: BaseKernel = LinearKernel(),
            inf_factor: Union[float, torch.Tensor, torch.nn.Parameter] = 1.0,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None
    ):
        self._core_module = KETKFModule(kernel=kernel, inf_factor=inf_factor)
        super().__init__(
            inf_factor=inf_factor,
            smoother=smoother,
            gpu=gpu,
            pre_transform=pre_transform,
            post_transform=post_transform
        )
        self.kernel = kernel

    def __str__(self):
        return 'Global KETKF(gamma={0}, kernel={1})'.format(
            str(self.inf_factor), str(self.kernel)
        )

    def __repr__(self):
        return 'KETKF({0},{1})'.format(
            repr(self.inf_factor), repr(self.kernel)
        )

    @property
    def inf_factor(self):
        return self._core_module.inf_factor

    @inf_factor.setter
    def inf_factor(self, new_factor):
        self._core_module = KETKFModule(
            inf_factor=new_factor, kernel=self.kernel
        )

    @property
    def kernel(self):
        return self._core_module.kernel

    @kernel.setter
    def kernel(self, new_kernel):
        self._core_module = KETKFModule(
            kernel=new_kernel, inf_factor=self.inf_factor
        )
