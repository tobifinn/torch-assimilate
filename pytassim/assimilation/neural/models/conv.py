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

# Internal modules


logger = logging.getLogger(__name__)


class PadCircular(torch.nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = torch.cat([x[..., -self.pad:], x, x[..., :self.pad]], dim=-1)
        return x


class DeepAssimilation(torch.nn.Module):
    def __init__(self):
        self.enc_prior = torch.nn.Sequential(
            self.conv_1d_layer(1, 8, stride=1), # 40
            self.conv_1d_layer(8, 16, stride=2), # 20
            self.conv_1d_layer(16, 16, stride=1), # 20
            self.conv_1d_layer(16, 32, stride=2), # 10
            self.conv_1d_layer(32, 32, stride=1), # 10
        )
        self.enc_obs = torch.nn.Sequential(
            torch.nn.Linear(8, 32,),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 128,),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
        )
        self.enc_combined = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.BatchNorm1d(20),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(20, 40),
        )
        self.obs_op = torch.nn.Sequential(
            torch.nn.Linear(40, 20),
            torch.nn.BatchNorm1d(20),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(20, 10),
            torch.nn.BatchNorm1d(10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, 5),
        )
        self.disc_prior = torch.nn.Sequential(
            torch.nn.Linear(45, 20),
            torch.nn.BatchNorm1d(20),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(20, 10),
            torch.nn.BatchNorm1d(10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid()
        )

    def encoder(self, prior, obs):
        lat_prior = self.enc_prior(prior)
        lat_obs = self.enc_obs(obs)
        lat_combined = lat_prior * lat_obs
        delta_latent = self.enc_combined(lat_combined)
        latent_state = prior + delta_latent
        return latent_state

    def decoder(self, latent_state):
        recon_obs = self.obs_op(latent_state)
        return recon_obs

    def forward(self, prior_ens_0, prior_ens_1, obs,):
        lat_prior = self.enc_prior(prior_ens_0)
        lat_obs = self.enc_obs(obs)
        lat_combined = lat_prior * lat_obs
        analysis = self.enc_combined(lat_combined)
        recon_obs = self.obs_op(analysis)

        crit_prior = self.disc_prior(
            torch.cat([obs, prior_ens_1], dim=-1)
        )
        crit_ana = self.disc_prior(
            torch.cat([obs, analysis], dim=-1)
        )
        return analysis, recon_obs, crit_prior, crit_ana

    def assimilate(self, back_state, obs, obs_cov):
        lat_prior = self.enc_prior(back_state)
        lat_obs = self.enc_obs(obs)
        lat_combined = lat_prior * lat_obs
        analysis = self.enc_combined(lat_combined)
        return analysis

    @staticmethod
    def conv_1d_layer(in_c, out_c, stride=1):
        return torch.nn.Sequential(
            PadCircular(1),
            torch.nn.Conv1d(in_c, out_c, kernel_size=3, padding=0, bias=False,
                            stride=stride),
            torch.nn.BatchNorm1d(out_c),
            torch.nn.LeakyReLU(),
        )
