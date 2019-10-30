#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 21.04.19
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
import torch.autograd

# Internal modules
from .standard import StandardDisc


logger = logging.getLogger(__name__)


class ZeroGradDisc(StandardDisc):
    def __init__(self, net, lam_fake=1, lam_real=1, anneal_rate=None):
        super().__init__(net)
        self.lam_fake = lam_fake
        self.lam_real = lam_real
        self.anneal_rate = anneal_rate

    def get_annealed_rate(self, it):
        if isinstance(self.anneal_rate, int):
            rate = 1 - it / self.anneal_rate
            return max(0, rate)
        else:
            return 1

    @staticmethod
    def _get_zero_reg(net_in, net_out):
        batch_size = net_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=net_out.sum(), inputs=net_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

    def _get_train_losses(self, real_data, fake_data, *args, **kwargs):
        batch_size = real_data.size()[0]

        real_data.requires_grad_()
        real_critic = self.forward(real_data, *args, **kwargs)
        real_labels = self.get_targets(batch_size, 0.0, real_data)
        real_loss = self.disc_loss(real_critic, real_labels)

        fake_data.requires_grad_()
        fake_critic = self.forward(fake_data, *args, **kwargs)
        fake_labels = self.get_targets(batch_size, 1.0, real_data)
        fake_loss = self.disc_loss(fake_critic, fake_labels)

        total_loss = real_loss + fake_loss

        if self.net.training:
            annealed_rate = self.get_annealed_rate(kwargs['it'])
            real_reg = torch.mean(self._get_zero_reg(real_data, real_critic))
            fake_reg = torch.mean(self._get_zero_reg(fake_data, fake_critic))

            reg_loss = annealed_rate * self.lam_real * real_reg
            reg_loss = reg_loss + annealed_rate * self.lam_fake * fake_reg
            total_loss = total_loss + reg_loss

        return total_loss, real_loss, fake_loss
