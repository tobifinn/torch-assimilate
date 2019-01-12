#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12/11/18
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
import torch

from sacred import Ingredient

# Internal modules
from pytassim.assimilation.neural.models.stoch_cycle import StochCycleGAN


logger = logging.getLogger(__name__)


model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    learning_rates = dict(
        gen=0.00001,
        disc_prior=0.00002,
        disc_obs=0.00002
    )
    lam = dict(
        recon=1,
        gan=1
    )
    disc_steps = 1
    obs_size = 20


class Model(StochCycleGAN):
    @model_ingredient.capture
    def __init__(self, obs_size, lam, disc_steps, _run):
        super().__init__(obs_size=obs_size)
        self.disc_steps = disc_steps
        self.lam = lam

    @model_ingredient.capture
    def get_optimizers(self, learning_rates):
        optimizers = {}
        for name, lr in learning_rates.items():
            optimizers[name] = torch.optim.Adam(self.params[name], lr=lr)
        return optimizers
