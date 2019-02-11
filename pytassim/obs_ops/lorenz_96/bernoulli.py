#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 1/25/19
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2019}  {Tobias Sebastian Finn}
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
import numpy as np

import torch

# Internal modules
from .identity import IdentityOperator


logger = logging.getLogger(__name__)


class BernoulliOperator(IdentityOperator):
    def __init__(self, shift=5, obs_points=None, len_grid=40,
                 random_state=None):
        super().__init__(obs_points=obs_points, len_grid=len_grid,
                         random_state=random_state)
        self.shift = shift

    @staticmethod
    def _np_sigmoid(state):
        return 1. / (1. + np.exp(-state))

    def obs_op(self, in_array, *args, **kwargs):
        obs_state = super().obs_op(in_array, *args, **kwargs)
        obs_state = obs_state - self.shift
        obs_state = self._np_sigmoid(obs_state)
        return obs_state

    def torch_operator(self):
        lin_layer = torch.nn.Linear(self.len_grid, len(self._sel_obs_points),
                                    bias=True)
        for param in lin_layer.parameters():
            param.requires_grad = False

        lin_layer.weight.data = torch.zeros_like(lin_layer.weight.data)
        for k, i in enumerate(self._sel_obs_points):
            lin_layer.weight.data[k, i] = 1.

        lin_layer.bias.data = torch.ones_like(lin_layer.bias.data)
        lin_layer.bias.data *= -self.shift
        operator = torch.nn.Sequential(
            lin_layer, torch.nn.Sigmoid()
        )
        return operator
