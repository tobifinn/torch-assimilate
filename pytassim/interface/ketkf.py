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
from typing import Union, Iterable, Callable

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
    """
    This is a kernelised verison of the `ensemble transform Kalman filter`
    :cite:`bishop_adaptive_2001`.
    This kernelised ensemble Kalman filter is a deterministic filter, where the
    state is globally updated. Ensemble weights are estimated in a reduced
    ensemble space, and then applied to a given state. This kernelised data
    assimilation can be used for non-linear observation operators. The
    observation operator is approximated by the ensemble and a given kernel.
    Furthermore, this implementation allows filtering in time and ensemble
    smoothing, similar to :cite:`hunt_four-dimensional_2004`. This ETKF
    implementation is less efficient for a linear kernel than
    :py:class:`pytassim.interface.filter.etkf.ETKF`, which should be
    then used instead.

    Parameters
    ----------
    kernel : child of :py:class:`pytassim.kernels.base_kernels.BaseKernel`
        This kernel is used to estimate the ensemble distance matrix.
        If no child of :py:class:`pytassim.kernels.base_kernels.BaseKernel`
        is used, the kernel should have atleast a :py:func:`forward` method.
    inf_factor : float, optional
        Multiplicative inflation factor :math:`\\rho``, which is applied to the
        background precision. An inflation factor greater one increases the
        ensemble spread, while a factor less one decreases the spread. Default
        is 1.0, which is the same as no inflation at all.
    smoother : bool, optional
        Indicates if this filter should be run in smoothing or in filtering
        mode. In smoothing mode, no analysis time is selected from given state
        and the ensemble weights are applied to the whole state. In filtering
        mode, the weights are applied only on selected analysis time. Default
        is False, indicating filtering mode.
    gpu : bool, optional
        Indicator if the weight estimation should be done on either GPU (True)
        or CPU (False): Default is None. For small models, estimation of the
        weights on CPU is faster than on GPU!.
    """
    def __init__(
            self,
            kernel: BaseKernel = LinearKernel(),
            inf_factor: Union[float, torch.Tensor, torch.nn.Parameter] = 1.0,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None,
            weight_save_path: Union[None, str] = None,
            forward_model: Union[None, Callable] = None
    ):
        self._core_module = KETKFModule(kernel=kernel)
        super().__init__(
            inf_factor=inf_factor,
            smoother=smoother,
            gpu=gpu,
            pre_transform=pre_transform,
            post_transform=post_transform,
            weight_save_path=weight_save_path,
            forward_model=forward_model
        )
        self.kernel = kernel

    def __str__(self):
        return 'Global KETKF(inf_factor={0}, kernel={1})'.format(
            str(self.inf_factor.item()), str(self.kernel)
        )

    def __repr__(self):
        return 'KETKF({0},{1})'.format(
            repr(self.inf_factor.item()), repr(self.kernel)
        )

    @property
    def inf_factor(self):
        return self._core_module.inf_factor

    @inf_factor.setter
    def inf_factor(self, new_factor):
        if isinstance(new_factor, (float, int)):
            new_factor = torch.tensor(new_factor, dtype=self.dtype)
        self._core_module = KETKFModule(
            inf_factor=new_factor, kernel=self.kernel
        )

    @property
    def kernel(self):
        return self._core_module.kernel

    @kernel.setter
    def kernel(self, new_kernel):
        new_kernel.to(dtype=self.dtype, device=self.device)
        self._core_module = KETKFModule(
            kernel=new_kernel, inf_factor=self.inf_factor
        )
