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
from torch import autograd

from sacred import Ingredient

# Internal modules
from pytassim.assimilation.neural.models.linear import DeepAssimilation


logger = logging.getLogger(__name__)


model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    learning_rates = dict(
        gen=0.0001,
        disc_prior=0.0002,
    )
    lam = dict(
        gen=1,
    )
    disc_steps = 1
    obs_size = 20
    lam_reg_disc = {
        'real': 10,
        'fake': 0
    }
    norm_spectral = {
        'gen': True,
        'disc': True
    }
    norm_batch = {
        'gen': False,
        'disc': False
    }


def compute_grad2(d_out, x_in):
    """
    From https://github.com/LMescheder/GAN_stability
    """
    x_in.requires_grad_()
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in, only_inputs=True, allow_unused=True,
        create_graph=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


class Model(DeepAssimilation):
    @model_ingredient.capture
    def __init__(self, lam_reg_disc, obs_size, disc_steps,
                 norm_spectral, norm_batch):
        super().__init__(obs_size=obs_size)
        self.disc_steps = disc_steps
        self.lam_reg_disc = lam_reg_disc
        self.optimizers = self.get_optimizers()
        self.recon_loss = torch.nn.MSELoss()

    @model_ingredient.capture
    def get_optimizers(self, learning_rates):
        optimizers = {}
        for name, lr in learning_rates.items():
            optimizers[name] = torch.optim.RMSprop(self.params[name], lr=lr)
        return optimizers

    def trainstep_disc(self, ens_prior_0, ens_prior_1, obs):
        self.train()
        batch_size = obs.size()[0]
        self.optimizers['disc_prior'].zero_grad()

        # Real data
        disc_in_real = torch.cat([ens_prior_1, obs], dim=-1,)
        disc_in_real.requires_grad_()

        crit_real = self.disc_prior(disc_in_real)
        real_targets = self.disc_prior.get_targets(
            batch_size, fill_val=1., tensor_type=crit_real
        )
        loss_real = self.disc_prior.disc_loss(crit_real, real_targets)
        if self.lam_reg_disc['real'] > 0:
            loss_real.backward(retain_graph=True)
            reg_real = compute_grad2(crit_real, disc_in_real).mean()
            loss_reg_real = self.lam_reg_disc['real'] / 2 * reg_real
            loss_reg_real.backward()
        else:
            loss_real.backward()
            loss_reg_real = torch.tensor((0.))

        # Fake data
        with torch.no_grad():
            analysis = self.encoder(ens_prior_0, obs)
        disc_in_fake = torch.cat([analysis, obs], dim=-1)
        disc_in_fake.requires_grad_()

        crit_fake = self.disc_prior(disc_in_fake)
        fake_targets = self.disc_prior.get_targets(
            batch_size, fill_val=0., tensor_type=crit_fake

        )
        loss_fake = self.disc_prior.disc_loss(crit_fake, fake_targets)
        if self.lam_reg_disc['fake'] > 0:
            loss_fake.backward(retain_graph=True)
            reg_fake = compute_grad2(crit_fake, disc_in_fake).mean()
            loss_reg_fake = self.lam_reg_disc['fake'] / 2 * reg_fake
            loss_reg_fake.backward()
        else:
            loss_fake.backward()
            loss_reg_fake = torch.tensor((0.))

        self.optimizers['disc_prior'].step()

        tot_loss = loss_real + loss_fake + loss_reg_real + loss_reg_fake
        losses = {
            'tot_loss': tot_loss,
            'loss_disc': loss_real + loss_fake,
            'loss_reg': loss_reg_real + loss_reg_fake,
            'loss_real': loss_real,
            'loss_fake': loss_fake,
            'loss_reg_real': loss_reg_real,
            'loss_reg_fake': loss_reg_fake
        }
        return losses

    def trainstep_gen(self, ens_prior_0, ens_prior_1, obs):
        self.train()
        batch_size = obs.size()[0]
        self.optimizers['gen'].zero_grad()

        analysis = self.encoder(ens_prior_0, obs)
        recon_obs = self.decoder(analysis)

        sum_dec_params = torch.sum(self.dec_net.weight.data)

        # Reconstruction loss
        loss_recon = self.recon_loss(recon_obs, obs)

        loss_recon.backward(retain_graph=True)

        # Adversarial loss
        disc_in = torch.cat([analysis, obs], dim=-1)
        crit_ana = self.disc_prior(disc_in)
        real_targets = self.disc_prior.get_targets(
            batch_size, 1.0, tensor_type=crit_ana
        )
        loss_adv = self.disc_prior.disc_loss(crit_ana, real_targets)
        loss_adv.backward(retain_graph=True)

        self.optimizers['gen'].step()

        tot_loss = loss_recon + loss_adv
        losses = {
            'tot_loss': tot_loss,
            'loss_recon': loss_recon,
            'loss_adv': loss_adv,
            'sum_decoder_params': sum_dec_params
        }
        return losses

    def eval_disc(self, ens_prior_0, ens_prior_1, obs):
        self.eval()
        batch_size = obs.size()[0]

        # Real data
        disc_in_real = torch.cat([ens_prior_1, obs], dim=-1,)
        disc_in_real.requires_grad_()

        crit_real = self.disc_prior(disc_in_real)
        real_targets = self.disc_prior.get_targets(
            batch_size, fill_val=1., tensor_type=crit_real
        )
        loss_real = self.disc_prior.disc_loss(crit_real, real_targets)

        # Fake data
        analysis = self.encoder(ens_prior_0, obs)
        disc_in_fake = torch.cat([analysis, obs], dim=-1)
        disc_in_fake.requires_grad_()

        crit_fake = self.disc_prior(disc_in_fake)
        fake_targets = self.disc_prior.get_targets(
            batch_size, fill_val=0., tensor_type=crit_fake

        )
        loss_fake = self.disc_prior.disc_loss(crit_fake, fake_targets)

        # Combine
        tot_loss = loss_real + loss_fake
        losses = {
            'tot_loss': tot_loss,
            'loss_real': loss_real,
            'loss_fake': loss_fake
        }
        return losses

    def eval_gen(self, ens_prior_0, ens_prior_1, obs):
        self.eval()
        batch_size = obs.size()[0]

        analysis = self.encoder(ens_prior_0, obs)
        recon_obs = self.decoder(analysis)

        # Reconstruction loss
        loss_recon = self.enc_net.recon_loss(recon_obs, obs)

        # Adversarial loss
        disc_in = torch.cat([analysis, obs], dim=-1)
        crit_ana = self.disc_prior(disc_in)
        real_targets = self.disc_prior.get_targets(
            batch_size, 1.0, tensor_type=crit_ana
        )
        loss_adv = self.disc_prior.disc_loss(crit_ana, real_targets)

        tot_loss = loss_recon + loss_adv
        losses = {
            'tot_loss': tot_loss,
            'loss_recon': loss_recon,
            'loss_adv': loss_adv
        }
        return losses
