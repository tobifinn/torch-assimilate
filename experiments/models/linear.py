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

# External modules
import torch

from sacred import Ingredient

# Internal modules


logger = logging.getLogger(__name__)


linear_ingredient = Ingredient('linear')


@linear_ingredient.config
def config():
    obs_size = 20
    grid_size = 40
    noise_size = 5
    hidden_size = (64, )


class Discriminator(torch.nn.Module):
    @linear_ingredient.capture
    def __init__(self, obs_size=20, grid_size=40, hidden_size=(64, )):
        super().__init__()

        curr_size = obs_size + grid_size
        layers = []
        for neurons in hidden_size:
            layers.append(torch.nn.Linear(curr_size, neurons, bias=False))
            layers.append(torch.nn.BatchNorm1d(neurons))
            layers.append(torch.nn.LeakyReLU())
            curr_size = neurons
        layers.append(torch.nn.Linear(curr_size, 1))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, model_state, observation, *args, **kwargs):
        disc_input = torch.cat([observation, model_state], dim=-1)
        disc_output = self.net(disc_input)
        return disc_output


class InferenceNet(torch.nn.Module):
    @linear_ingredient.capture
    def __init__(self,  obs_size=20, grid_size=40, noise_size=5,
                 hidden_size=(64, )):
        super().__init__()

        curr_size = obs_size + grid_size
        layers = []
        for neurons in hidden_size:
            layers.append(torch.nn.Linear(curr_size, neurons, bias=False))
            layers.append(torch.nn.BatchNorm1d(neurons))
            layers.append(torch.nn.LeakyReLU())
            curr_size = neurons
        layers.append(torch.nn.Linear(curr_size, 40))

        self.data_net = torch.nn.Sequential(*layers)
        self.noise_net = torch.nn.Sequential(
            torch.nn.Linear(noise_size, 64, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 40)
        )

    def gen_rand_data(self, batch_size, tensor_type=None):
        rand_vec = torch.normal(
            torch.zeros((batch_size, self.noise_size)),
            torch.ones((batch_size, self.noise_size)),
        )
        if tensor_type is not None:
            rand_vec = rand_vec.to(tensor_type)
        return rand_vec

    def forward(self, observation, prior, *args, **kwargs):
        batch_size = observation.size()[0]
        noise = self.gen_rand_data(batch_size, observation)

        data_in = torch.cat([observation, prior], dim=-1)
        data_out = self.data_net(data_in)
        noise_out = self.noise_net(noise)
        delta_ana = data_out * noise_out

        analysis = prior + delta_ana
        return analysis

    def assimilate(self, in_state, obs, obs_cov):
        back_state = in_state.view(-1, self.grid_size)
        obs = obs.view(1, -1).expand(back_state.size()[0], -1)
        analysis = self.forward(observation=obs, prior=back_state)
        analysis = analysis.view_as(in_state)
        analysis = analysis.detach()
        return analysis
