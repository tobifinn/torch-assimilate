#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 1/11/19
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
import sys
import os

# External modules
import torch

from sacred import Ingredient

# Internal modules
sys.path.append(
    os.path.join(os.path.dirname(__file__), '../..', 'experiments')
)

from experiments.models.linear import Discriminator, InferenceNet
from pytassim.toolbox import Autoencoder, LossWrapper, StandardDisc


logger = logging.getLogger(__name__)


model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    learning_rates = dict(
        gen=0.0001,
        disc=0.0001,
    )
    obs_size = 8
    grid_size = 40
    noise_size = 5
    hidden_size = (64, )


@model_ingredient.capture
def get_models(dataset, learning_rates, obs_size, grid_size, noise_size,
               hidden_size):
    obs_operator = dataset.obs_operator.torch_operator()
    inference_net = InferenceNet(obs_size=obs_size, grid_size=grid_size,
                                 noise_size=noise_size, hidden_size=hidden_size)
    disc_net = Discriminator(obs_size=obs_size, grid_size=grid_size,
                             hidden_size=hidden_size)
    discriminator = StandardDisc(net=disc_net)
    discriminator.optimizer = torch.optim.RMSprop(
        discriminator.trainable_params, lr=learning_rates['disc']
    )

    autoencoder = Autoencoder(inference_net=inference_net,
                              obs_operator=obs_operator)
    autoencoder.optimizer = torch.optim.RMSprop(autoencoder.trainable_params,
                                                lr=learning_rates['gen'])
    autoencoder.recon_loss = LossWrapper(torch.nn.BCELoss())
    autoencoder.back_loss = discriminator

    return autoencoder, discriminator
