#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 18.08.20
#
# Created for torch-assimilate
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
from typing import Type, Union, Tuple, Iterable

# External modules
import torch

# Internal modules
from .ketkf_core import KETKFAnalyser
from .etkf_core import CorrMixin, UnCorrMixin
from .etkf import ETKFBase

from pytassim.kernels.base_kernels import BaseKernel
from pytassim.transform.base import BaseTransformer


logger = logging.getLogger(__name__)


__all__ = ['KETKFCorr', 'KETKFUncorr']


class KETKFBase(ETKFBase):
    """
    The base class for the kernelized ensemble transform Kalman filter.
    """
    def __init__(
            self,
            kernel: Type[BaseKernel],
            inf_factor: Union[torch.Tensor, float, torch.nn.Parameter] = 1.0,
            smoother: bool = False, gpu: bool = False,
            pre_transform: Union[None, Iterable[Type[BaseTransformer]]] = None,
            post_transform: Union[None, Iterable[Type[BaseTransformer]]] = None
    ):
        self._analyser = KETKFAnalyser(kernel=kernel, inf_factor=inf_factor)
        super().__init__(inf_factor=inf_factor, smoother=smoother, gpu=gpu,
                         pre_transform=pre_transform,
                         post_transform=post_transform)
        self._name = 'Global Kernel ETKF'

    def __str__(self):
        return 'Kernel ETKF ({0:s}, {1})'.format(str(self.kernel),
                                                 self.inf_factor)

    def __repr__(self):
        return 'KETKF({0:s})'.format(repr(self.kernel))

    @property
    def inf_factor(self) -> Union[float, torch.Tensor, torch.nn.Parameter]:
        return self._analyser.inf_factor

    @inf_factor.setter
    def inf_factor(
            self,
            new_factor: Union[float, torch.Tensor, torch.nn.Parameter]
    ):
        self._analyser = KETKFAnalyser(
            kernel=self._analyser.kernel, inf_factor=new_factor
        )

    @property
    def kernel(self) -> Type[BaseKernel]:
        return self._analyser.kernel

    @kernel.setter
    def kernel(self, new_kernel: Type[BaseKernel]):
        self._analyser = KETKFAnalyser(
            kernel=new_kernel, inf_factor=self._analyser.inf_factor
        )


class KETKFCorr(CorrMixin, KETKFBase):
    """
    This is a kernelised verison of the `ensemble transform Kalman filter`
    :cite:`bishop_adaptive_2001` for correlated observation covariances.
    This kernelised ensemble Kalman filter is a deterministic filter, where the
    state is globally updated. Ensemble weights are estimated in a reduced
    ensemble space, and then applied to a given state. This kernelised data
    assimilation can be used for non-linear observation operators. The
    observation operator is approximated by the ensemble and a given kernel.
    Furthermore, this implementation allows filtering in time and ensemble
    smoothing, similar to :cite:`hunt_four-dimensional_2004`. This ETKF
    implementation is less efficient for a linear kernel than
    :py:class:`pytassim.assimilation.filter.etkf.ETKFCorr`, which should be
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
    def __str__(self):
        return 'Correlated {0:s}'.format(str(super(KETKFBase)))

    def __repr__(self):
        return 'Corr{0:s}'.format(repr(super(KETKFBase)))


class KETKFUncorr(UnCorrMixin, KETKFBase):
    """
    This is a kernelised verison of the `ensemble transform Kalman filter`
    :cite:`bishop_adaptive_2001` for uncorrelated observation covariances.
    This kernelised ensemble Kalman filter is a deterministic filter, where the
    state is globally updated. Ensemble weights are estimated in a reduced
    ensemble space, and then applied to a given state. This kernelised data
    assimilation can be used for non-linear observation operators. The
    observation operator is approximated by the ensemble and a given kernel.
    Furthermore, this implementation allows filtering in time and ensemble
    smoothing, similar to :cite:`hunt_four-dimensional_2004`. This ETKF
    implementation is less efficient for a linear kernel than
    :py:class:`pytassim.assimilation.filter.etkf.ETKFUncorr`, which should be
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
    def __str__(self):
        return 'Uncorrelated {0:s}'.format(str(super(KETKFBase)))

    def __repr__(self):
        return 'Uncorr{0:s}'.format(repr(super(KETKFBase)))
