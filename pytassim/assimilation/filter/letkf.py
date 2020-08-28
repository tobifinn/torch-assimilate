#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12/7/18
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2018}  {Tobias Sebastian Finn}
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
import xarray as xr
from tqdm import tqdm
import torch

# Internal modules
from .etkf import ETKFBase
from .etkf_core import CorrMixin, UnCorrMixin
from .letkf_core import LETKFAnalyser

from pytassim.localization.localization import BaseLocalization
from pytassim.transform.base import BaseTransformer

logger = logging.getLogger(__name__)


class LETKFBase(ETKFBase):
    """
    The base class for the localised ensemble transform Kalman filter.
    """
    def __init__(
            self,
            localization: Union[None, BaseLocalization] = None,
            inf_factor: Union[torch.Tensor, float, torch.nn.Parameter] = 1.0,
            smoother: bool = False, gpu: bool = False,
            pre_transform: Union[None, Iterable[Type[BaseTransformer]]] = None,
            post_transform: Union[None, Iterable[Type[BaseTransformer]]] = None
    ):
        self._analyser = LETKFAnalyser(localization=localization,
                                       inf_factor=inf_factor)
        super().__init__(inf_factor=inf_factor, smoother=smoother, gpu=gpu,
                         pre_transform=pre_transform,
                         post_transform=post_transform)
        self._analyser = LETKFAnalyser(localization=localization,
                                       inf_factor=inf_factor)
        self._name = 'Sequential LETKF'

    def __str__(self):
        return '{0:s}({1:s}, {2})'.format(self._name, str(self.localization),
                                          self.inf_factor)

    def __repr__(self):
        return 'SeqLETKF({0:s})'.format(repr(self.localization))

    @property
    def localization(self) -> Type[BaseLocalization]:
        return self._analyser.localization

    @localization.setter
    def localization(self, new_locs: Type[BaseLocalization]):
        """
        Sets a new localization.
        """
        self._analyser = LETKFAnalyser(
            localization=new_locs, inf_factor=self.analyser.inf_factor
        )

    @property
    def inf_factor(self) -> Union[float, torch.Tensor, torch.nn.Parameter]:
        return self._analyser.inf_factor

    @inf_factor.setter
    def inf_factor(
            self,
            new_factor: Union[float, torch.Tensor, torch.nn.Parameter]
    ):
        """
        Sets a new inflation factor.
        """
        if self.analyser is None:
            localization = None
        else:
            localization = self.analyser.localization
        self._analyser = LETKFAnalyser(
            localization=localization, inf_factor=new_factor
        )


class LETKFCorr(CorrMixin, LETKFBase):
    """
    This is an implementation of the `localized ensemble transform Kalman
    filter` :cite:`hunt_efficient_2007` for correlated observations.
    This is a localized version of the `ensemble transform Kalman filter`
    :cite:`bishop_adaptive_2001`. This method iterates independently over each
    grid point in given background state. Given localization instance can be
    used to
    constrain the influence of observations in space. The ensemble weights are
    calculated for every grid point and independently applied to every grid
    point. This implementation follows :cite:`hunt_efficient_2007`, with local
    weight estimation and is implemented in PyTorch. This implementation allows
    filtering in time based on linear propagation assumption
    :cite:`hunt_four-dimensional_2004` and ensemble smoothing.

    Parameters
    ----------
    smoothing : bool, optional
        Indicates if this filter should be run in smoothing or in filtering
        mode. In smoothing mode, no analysis time is selected from given state
        and the ensemble weights are applied to the whole state. In filtering
        mode, the weights are applied only on selected analysis time. Default
        is False, indicating filtering mode.
    localization : obj or None, optional
        This localization is used to localize and constrain observations
        spatially. If this localization is None, no localization is applied such
        it is an inefficient version of the `ensemble transform Kalman filter`.
        Default value is None, indicating no localization at all.
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
    def __str__(self):
        return 'Correlated {0:s}'.format(str(super(LETKFBase)))

    def __repr__(self):
        return 'Corr{0:s}'.format(repr(super(LETKFBase)))


class LETKFUncorr(UnCorrMixin, LETKFBase):
    """
    This is an implementation of the `localized ensemble transform Kalman
    filter` :cite:`hunt_efficient_2007` for uncorrelated observations.
    This is a localized version of the `ensemble transform Kalman filter`
    :cite:`bishop_adaptive_2001`. This method iterates independently over each
    grid point in given background state. Given localization instance can be
    used to
    constrain the influence of observations in space. The ensemble weights are
    calculated for every grid point and independently applied to every grid
    point. This implementation follows :cite:`hunt_efficient_2007`, with local
    weight estimation and is implemented in PyTorch. This implementation allows
    filtering in time based on linear propagation assumption
    :cite:`hunt_four-dimensional_2004` and ensemble smoothing.

    Parameters
    ----------
    smoothing : bool, optional
        Indicates if this filter should be run in smoothing or in filtering
        mode. In smoothing mode, no analysis time is selected from given state
        and the ensemble weights are applied to the whole state. In filtering
        mode, the weights are applied only on selected analysis time. Default
        is False, indicating filtering mode.
    localization : obj or None, optional
        This localization is used to localize and constrain observations
        spatially. If this localization is None, no localization is applied such
        it is an inefficient version of the `ensemble transform Kalman filter`.
        Default value is None, indicating no localization at all.
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
    def __str__(self):
        return 'Uncorrelated {0:s}'.format(str(super(LETKFBase)))

    def __repr__(self):
        return 'Uncorr{0:s}'.format(repr(super(LETKFBase)))

