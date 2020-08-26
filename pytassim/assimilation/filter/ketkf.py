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

# External modules

# Internal modules
from .ketkf_core import KETKFAnalyser
from .etkf_core import CorrMixin, UnCorrMixin
from .etkf import ETKFBase


logger = logging.getLogger(__name__)


class KETKFBase(ETKFBase):
    def __init__(self, kernel, inf_factor=1.0, smoother=True, gpu=False,
                 pre_transform=None, post_transform=None):
        self._analyser = KETKFAnalyser(kernel=kernel, inf_factor=inf_factor)
        super().__init__(inf_factor=inf_factor, smoother=smoother, gpu=gpu,
                         pre_transform=pre_transform,
                         post_transform=post_transform)
        self._name = 'Global Kernel ETKF'

    @property
    def inf_factor(self):
        return self._analyser.inf_factor

    @inf_factor.setter
    def inf_factor(self, new_factor):
        self._analyser = KETKFAnalyser(
            kernel=self._analyser.kernel, inf_factor=new_factor
        )

    @property
    def kernel(self):
        return self._analyser.kernel

    @kernel.setter
    def kernel(self, new_kernel):
        self._analyser = KETKFAnalyser(
            kernel=new_kernel, inf_factor=self._analyser.inf_factor
        )

    def _normalise_obs(self, pseudo_obs, obs, cinv):
        normed_perts = self._mul_cinv(pseudo_obs, cinv)
        normed_obs = self._mul_cinv(obs.view(1, -1), cinv)
        return normed_perts, normed_obs


class KETKFCorr(CorrMixin, KETKFBase):
    pass


class KETKFUncorr(UnCorrMixin, KETKFBase):
    """
    This is an implementation of the `ensemble transform Kalman filter`
    :cite:`bishop_adaptive_2001` for uncorrelated observation covariances.
    This ensemble Kalman filter is a deterministic filter, where the state is
    update globally. This ensemble Kalman filter estimates ensemble weights in
    weight space, which are then applied to the given state. This implementation
    follows :cite:`hunt_efficient_2007` with global weight estimation and is
    implemented in PyTorch.
    This implementation allows filtering in time based on linear propagation
    assumption :cite:`hunt_four-dimensional_2004` and ensemble smoothing.

    Parameters
    ----------
    smoother : bool, optional
        Indicates if this filter should be run in smoothing or in filtering
        mode. In smoothing mode, no analysis time is selected from given state
        and the ensemble weights are applied to the whole state. In filtering
        mode, the weights are applied only on selected analysis time. Default
        is False, indicating filtering mode.
    inf_factor : float, optional
        Multiplicative inflation factor :math:`\\rho``, which is applied to the
        background precision. An inflation factor greater one increases the
        ensemble spread, while a factor less one decreases the spread. Default
        is 1.0, which is the same as no inflation at all.
    gpu : bool, optional
        Indicator if the weight estimation should be done on either GPU (True)
        or CPU (False): Default is None. For small models, estimation of the
        weights on CPU is faster than on GPU!.
    """
    pass
