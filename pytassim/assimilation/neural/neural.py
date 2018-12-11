#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 26.03.18
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

# External modules

# Internal modules
from ..base import BaseAssimilation


logger = logging.getLogger(__name__)


class NeuralAssimilation(BaseAssimilation):
    """
    NeuralAssimilation is a class to assimilate observations into a state with
    neural networks. This class supports PyTorch natively. The ``assimilate``
    method of a given :py:class:`torch.nn.Module` is used to to assimilate the
    given state. No observation operator is needed for given observations.

    Parameters
    ----------
    module : child of :py:class:`torch.nn.Module`
        This module is used to assimilate given observations into given state.
        The module needs an ``assimilate`` method, where state, flattened
        observations and flattened observation covariance is given.
    gpu : bool, optional
        This boolean indicates if the assimilation should be done on gpu (True)
        or cpu (False). Default value is False, indicating computations on cpu.
    """
    def __init__(self, module, gpu=False):
        super().__init__(gpu=gpu)
        self.module = module

    def update_state(self, state, observations, analysis_time):
        obs_state, obs_cov, _ = self._prepare_obs(observations)
        prepared_torch = self._states_to_torch(state.values, obs_state, obs_cov)
        torch_analysis = self.module.assimilate(*prepared_torch)
        if self.gpu:
            torch_analysis = torch_analysis.cpu()
        analysis = state.copy(deep=True, data=torch_analysis.numpy())
        return analysis
