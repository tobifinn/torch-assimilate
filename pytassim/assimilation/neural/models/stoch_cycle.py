#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12/13/18
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


class StochCycleGAN(torch.nn.Module):
    def __init__(self, grid_size=40, obs_size=8, gen_dims=(64, ),
                 rand_dims=(64, ), disc_dims=(64, )):
        super().__init__()
        self.grid_size = grid_size
        self.obs_size = obs_size
        self.gen_dims = gen_dims
        self.rand_dims = rand_dims
        self.disc_dims = disc_dims

        self.inference = StochBlock(
            grid_size+obs_size, (grid_size, obs_size), hidden_size=64,
            data_dims=gen_dims, rand_dims=rand_dims
        )

        self.obs_operator = StochBlock(
            grid_size, (obs_size, grid_size), hidden_size=64,
            data_dims=gen_dims, rand_dims=rand_dims
        )

        self.disc_prior = Discriminator(
            grid_size + obs_size, hidden_size=64, disc_dims=disc_dims
        )

        self.disc_obs = Discriminator(
            grid_size + obs_size, hidden_size=64, disc_dims=disc_dims
        )
        self.rand_x = None
        self.rand_y = None
        self.batch_size = None
        self.params = dict(
            gen=[p for n, p in self.named_parameters() if 'disc' not in n],
            disc_prior=[p for n, p in self.named_parameters()
                        if 'disc_prior' in n],
            disc_obs=[p for n, p in self.named_parameters()
                        if 'disc_obs' in n],
        )

    def forward(self, prior_ens_0, prior_ens_1, obs):
        self.batch_size = obs.size()[0]
        self.rand_x = torch.normal(
            mean=torch.zeros((self.batch_size, self.grid_size)),
            std=torch.ones((self.batch_size, self.grid_size))
        ).cuda()
        self.rand_y = torch.normal(
            mean=torch.zeros((self.batch_size, self.obs_size)),
            std=torch.ones((self.batch_size, self.obs_size))
        ).cuda()

        # Forward inference
        pre_inference = torch.cat([prior_ens_0, obs], dim=1)
        del_gen_state_x, gen_rand_y = self.inference(pre_inference, self.rand_x)
        gen_state_x = prior_ens_0 + del_gen_state_x
        recon_obs, recon_rand_x = self.obs_operator(gen_state_x, self.rand_y)

        # Backward ensemble encoding
        gen_obs_y, gen_rand_x = self.obs_operator(prior_ens_1, self.rand_y)
        gen_prior_inference = torch.cat([prior_ens_0, gen_obs_y], dim=1)
        del_recon_prior, recon_rand_y = self.inference(gen_prior_inference,
                                                       self.rand_x)
        recon_prior = prior_ens_0 + del_recon_prior

        return gen_state_x, gen_rand_y, recon_obs, recon_rand_x, gen_obs_y,\
               gen_rand_x, recon_prior, recon_rand_y

    def assimilate(self, first_guess, obs, obs_cov):
        first_guess_reshaped = first_guess.view(-1, 40)
        ens_size = first_guess_reshaped.size()[0]

        obs = obs.view(1, -1).expand(ens_size, -1)
        rand_in = torch.normal(
            mean=torch.zeros((ens_size, self.grid_size)),
            std=torch.ones((ens_size, self.grid_size))
        ).type(first_guess.type()).to(first_guess.device)
        pre_inference = torch.cat([first_guess_reshaped, obs], dim=1)
        del_ana, _ = self.inference(pre_inference, rand_in)
        analysis = first_guess + del_ana
        analysis = analysis.view_as(first_guess)
        analysis = analysis.detach()
        return analysis

    def loss_gen_forward(self, prior_ens_0, prior_ens_1, obs,
                             lam_recon=1, lam_gan=1):

        forward_states = self.forward(prior_ens_0, prior_ens_1, obs)
        gen_state_x, gen_rand_y, recon_obs, recon_rand_x = forward_states[:4]

        loss_recon_obs = self.obs_operator.recon_loss(recon_obs, obs)
        loss_recon_noise = self.inference.recon_loss(recon_rand_x, self.rand_x)

        loss_recon = loss_recon_obs + loss_recon_noise

        disc_prior_in = torch.cat([gen_state_x, obs], dim=1)
        crit_state_x = self.disc_prior(disc_prior_in)
        targets = self.disc_prior.get_targets(
            self.batch_size, fill_val=1.0, dtype=prior_ens_0.dtype,
            device=prior_ens_0.device
        )
        loss_disc_state_x = self.disc_prior.disc_loss(crit_state_x, targets)

        disc_obs_in = torch.cat([prior_ens_1, gen_rand_y], dim=1)
        crit_gen_rand_y = self.disc_obs(disc_obs_in)
        targets = self.disc_obs.get_targets(
            self.batch_size, fill_val=1.0, dtype=prior_ens_0.dtype,
            device=prior_ens_0.device
        )
        loss_disc_gen_rand_y = self.disc_obs.disc_loss(crit_gen_rand_y, targets)

        loss_gan = loss_disc_state_x + loss_disc_gen_rand_y

        tot_loss = lam_recon * loss_recon + lam_gan * loss_gan

        losses = {
            'tot_loss': tot_loss,
            'loss_recon': loss_recon,
            'loss_gan': loss_gan,
            'loss_recon_obs': loss_recon_obs,
            'loss_recon_noise': loss_recon_noise,
            'loss_disc_state_x': loss_disc_state_x,
            'loss_disc_gen_rand_y': loss_disc_gen_rand_y
        }
        return losses, forward_states

    def loss_gen_backward(self, prior_ens_0, prior_ens_1, obs,
                          lam_recon=1, lam_gan=1):

        forward_states = self.forward(prior_ens_0, prior_ens_1, obs)
        gen_obs_y, gen_rand_x, recon_prior, recon_rand_y = forward_states[4:]

        loss_recon_prior = self.inference.recon_loss(recon_prior, prior_ens_1)
        loss_recon_noise = self.obs_operator.recon_loss(recon_rand_y,
                                                        self.rand_y)
        loss_recon = loss_recon_prior + loss_recon_noise

        disc_prior_in = torch.cat([gen_rand_x, obs], dim=1)
        crit_gen_rand_x = self.disc_prior(disc_prior_in)
        targets = self.disc_prior.get_targets(
            self.batch_size, fill_val=1.0, dtype=prior_ens_0.dtype,
            device=prior_ens_0.device
        )
        loss_disc_gen_rand_x = self.disc_prior.disc_loss(crit_gen_rand_x,
                                                         targets)

        disc_obs_in = torch.cat([prior_ens_1, gen_obs_y], dim=1)
        crit_obs_y = self.disc_obs(disc_obs_in)
        targets = self.disc_obs.get_targets(
            self.batch_size, fill_val=1.0, dtype=prior_ens_0.dtype,
            device=prior_ens_0.device
        )
        loss_disc_obs_y = self.disc_obs.disc_loss(crit_obs_y, targets)

        loss_gan = loss_disc_gen_rand_x + loss_disc_obs_y

        tot_loss = lam_recon * loss_recon + lam_gan * loss_gan

        losses = {
            'tot_loss': tot_loss,
            'loss_recon': loss_recon,
            'loss_gan': loss_gan,
            'loss_recon_prior': loss_recon_prior,
            'loss_recon_noise': loss_recon_noise,
            'loss_disc_gen_rand_x': loss_disc_gen_rand_x,
            'loss_disc_obs_y': loss_disc_obs_y
        }
        return losses, forward_states

    def loss_disc_prior(self, prior_ens_0, prior_ens_1, obs):
        forward_states = self.forward(prior_ens_0, prior_ens_1, obs)

        fake_targets = self.disc_prior.get_targets(
            self.batch_size, fill_val=0.0, dtype=prior_ens_0.dtype,
            device=prior_ens_0.device
        )
        real_targets = self.disc_prior.get_targets(
            self.batch_size, fill_val=1.0, dtype=prior_ens_0.dtype,
            device=prior_ens_0.device
        )
        gen_state_x, gen_rand_x = [forward_states[i] for i in [0, 5]]

        in_rand_fake = torch.cat([gen_rand_x, self.rand_y], dim=1)
        crit_rand_fake = self.disc_prior(in_rand_fake)
        loss_rand_fake = self.disc_prior.disc_loss(crit_rand_fake,
                                                   fake_targets)

        in_rand_real = torch.cat([self.rand_x, self.rand_y], dim=1)
        crit_rand_real = self.disc_prior(in_rand_real)
        loss_rand_real = self.disc_prior.disc_loss(crit_rand_real,
                                                   real_targets)

        loss_rand = loss_rand_fake + loss_rand_real

        in_state_fake = torch.cat([gen_state_x, obs], dim=1)
        crit_state_fake = self.disc_prior(in_state_fake)
        loss_state_fake = self.disc_prior.disc_loss(crit_state_fake,
                                                    fake_targets)

        in_state_real = torch.cat([prior_ens_1, obs], dim=1)
        crit_state_real = self.disc_prior(in_state_real)
        loss_state_real = self.disc_prior.disc_loss(crit_state_real,
                                                    real_targets)

        loss_state = loss_state_fake + loss_state_real

        tot_loss = loss_rand + loss_state

        losses = {
            'tot_loss': tot_loss,
            'loss_state': loss_state,
            'loss_rand': loss_rand,
            'loss_rand_fake': loss_rand_fake,
            'loss_rand_real': loss_rand_real,
            'loss_state_fake': loss_state_fake,
            'loss_state_real': loss_state_real,
        }
        return losses, forward_states

    def loss_disc_obs(self, prior_ens_0, prior_ens_1, obs):
        forward_states = self.forward(prior_ens_0, prior_ens_1, obs)

        fake_targets = self.disc_obs.get_targets(
            self.batch_size, fill_val=0.0, dtype=prior_ens_0.dtype,
            device=prior_ens_0.device
        )
        real_targets = self.disc_obs.get_targets(
            self.batch_size, fill_val=1.0, dtype=prior_ens_0.dtype,
            device=prior_ens_0.device
        )
        gen_obs_y, gen_rand_y = [forward_states[i] for i in [4, 1]]

        in_rand_fake = torch.cat([self.rand_x, gen_rand_y], dim=1)
        crit_rand_fake = self.disc_obs(in_rand_fake)
        loss_rand_fake = self.disc_obs.disc_loss(crit_rand_fake,
                                                 fake_targets)

        in_rand_real = torch.cat([self.rand_x, self.rand_y], dim=1)
        crit_rand_real = self.disc_obs(in_rand_real)
        loss_rand_real = self.disc_obs.disc_loss(crit_rand_real,
                                                 real_targets)

        loss_rand = loss_rand_fake + loss_rand_real

        in_state_fake = torch.cat([prior_ens_1, gen_obs_y], dim=1)
        crit_state_fake = self.disc_obs(in_state_fake)
        loss_state_fake = self.disc_obs.disc_loss(crit_state_fake,
                                                  fake_targets)

        in_state_real = torch.cat([prior_ens_1, obs], dim=1)
        crit_state_real = self.disc_obs(in_state_real)
        loss_state_real = self.disc_obs.disc_loss(crit_state_real,
                                                  real_targets)

        loss_state = loss_state_fake + loss_state_real

        tot_loss = loss_state + loss_rand

        losses = {
            'tot_loss': tot_loss,
            'loss_state': loss_state,
            'loss_rand': loss_rand,
            'loss_rand_fake': loss_rand_fake,
            'loss_rand_real': loss_rand_real,
            'loss_state_fake': loss_state_fake,
            'loss_state_real': loss_state_real,
        }
        return losses, forward_states
