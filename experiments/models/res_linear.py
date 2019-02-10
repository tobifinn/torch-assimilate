#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 2/8/19
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
import torch.nn

# Internal modules


logger = logging.getLogger(__name__)


class ResidualFiLM(torch.nn.Module):
    def __init__(self, in_size, hidden_size, batch_norm=True):
        super().__init__()
        if batch_norm:
            self.lin1 = torch.nn.Sequential(
                torch.nn.Linear(in_size, hidden_size, bias=False),
                torch.nn.BatchNorm1d(hidden_size),
                torch.nn.LeakyReLU(),
            )
        else:
            self.lin1 = torch.nn.Sequential(
                torch.nn.Linear(in_size, hidden_size, bias=True),
                torch.nn.LeakyReLU()
            )
        self.lin2 = torch.nn.Linear(hidden_size, in_size, bias=True)

    def forward(self, x, gamma=None, beta=None):
        res_layer = self.lin1(x)
        if gamma is not None:
            res_layer = gamma * res_layer
        if beta is not None:
            res_layer = res_layer + beta
        res_layer = self.lin2(res_layer)
        out_layer = res_layer + x
        return out_layer


class ResidualInferenceNet(torch.nn.Module):
    def __init__(self,  obs_size=20, grid_size=40, noise_size=5,
                 hidden_size=(64, )):
        super().__init__()
        self.noise_size = noise_size
        self.grid_size = grid_size
        self.obs_size = obs_size

        self.obs_features = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU()
        )
        self.films_layers = []
        self.res_blocks = []
        curr_res_in = grid_size
        for size in hidden_size:
            gamma = torch.nn.Linear(64, size, bias=True)
            beta = torch.nn.Linear(64, size, bias=True)
            res_block = ResidualFiLM(curr_res_in, size)
            self.films_layers.append((gamma, beta))
            self.res_blocks.append(res_block)
            curr_res_in = size

        self.noise_feature = torch.nn.Sequential(
            torch.nn.Linear(noise_size, 64, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(),
        )
        self.noise_gamma = torch.nn.Linear(64, hidden_size[-1], bias=False)
        self.noise_beta = torch.nn.Linear(64, hidden_size[-1], bias=False)

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
        noise_features = self.noise_feature(noise)

        obs_features = self.obs_features(observation)

        analysis = prior
        for i, res_block in self.res_blocks[:-1]:
            obs_gamma = self.films_layers[i][0](obs_features)
            obs_beta = self.films_layers[i][1](obs_features)
            analysis = res_block(analysis, gamma=obs_gamma, beta=obs_beta)

        obs_gamma = self.films_layers[-1][0](obs_features)
        obs_beta = self.films_layers[-1][1](obs_features)
        noise_gamma = self.noise_gamma(noise_features)
        noise_beta = self.noise_beta(noise_features)
        last_gamma = noise_gamma * obs_gamma
        last_beta = noise_beta + obs_beta
        analysis = self.res_blocks[-1](analysis, gamma=last_gamma,
                                       beta=last_beta)
        return analysis

    def assimilate(self, in_state, obs, obs_cov):
        back_state = in_state.view(-1, self.grid_size)
        obs = obs.view(1, -1).expand(back_state.size()[0], -1)
        analysis = self.forward(observation=obs, prior=back_state)
        analysis = analysis.view_as(in_state)
        analysis = analysis.detach()
        return analysis
