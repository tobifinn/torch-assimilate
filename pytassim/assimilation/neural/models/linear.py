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
from .stoch_block import StochBlock, Discriminator


logger = logging.getLogger(__name__)


class DeepAssimilation(torch.nn.Module):
    def __init__(self, obs_size=8, grid_size=40,
                 norm_spectral={'gen': True, 'disc': True},
                 norm_batch={'gen': True, 'disc': True}):
        super().__init__()
        self.obs_size = obs_size
        self.grid_size = grid_size
        self.enc_net = StochBlock(
            in_size=grid_size + obs_size,
            out_size=(grid_size, obs_size),
            data_dims=(64, 32, 16, 32, 64),
            spectral_norm=norm_spectral['gen'],
            batch_norm=norm_batch['gen']
        )
        self.dec_net = torch.nn.Linear(grid_size, obs_size, bias=False)
        self.dec_net.weight.data = torch.zeros_like(self.dec_net.weight.data)
        self.disc_prior = Discriminator(
            in_size=grid_size + obs_size,
            disc_dims=(64, 64, 64, 32, 32, 32, 16, 16,),
            spectral_norm=norm_spectral['disc'],
            batch_norm=norm_batch['disc']
        )
        self.params = dict(
            gen=[p for n, p in self.named_parameters() if 'enc_net' in n],
            disc_prior=[p for n, p in self.named_parameters()
                        if 'disc_prior' in n],
        )

    def encoder(self, prior, obs):
        rand_data = self.enc_net.gen_rand_data(
            prior.size()[0], tensor_type=prior)
        concat_enc_in = torch.cat([prior, obs], dim=-1)
        delta_latent, _ = self.enc_net(concat_enc_in, rand_data)
        latent_state = prior + delta_latent
        return latent_state

    def decoder(self, latent_state):
        recon_obs = self.dec_net(latent_state)
        return recon_obs

    def forward(self, ens_prior_0, ens_prior_1, obs):
        analysis = self.encoder(ens_prior_0, obs)
        recon_obs = self.decoder(analysis)
        return analysis, recon_obs

    def assimilate(self, in_state, obs, obs_cov):
        back_state = in_state.view(-1, self.grid_size)
        obs = obs.view(1, -1).expand(back_state.size()[0], -1)
        analysis = self.encoder(back_state, obs)
        analysis = analysis.view_as(in_state)
        analysis = analysis.detach()
        return analysis
