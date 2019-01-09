#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 1/9/19
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

# Internal modules
from ..linear_net import create_net


logger = logging.getLogger(__name__)


class Discriminator(torch.nn.Module):
    """
    Standard discriminator for generative adversarial networks. This
    discriminator
    """
    def __init__(self, in_size, out_size=1, hidden_size=16, disc_dims=(64, ),
                 batch_norm=False):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.disc_dims = disc_dims
        self.batch_norm = batch_norm

        self.loss_func = torch.nn.BCEWithLogitsLoss()

        self.net = create_net(self.in_size, hidden_size, disc_dims, batch_norm)
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, out_size),
        )

    def get_targets(self, batch_size, fill_val=1.0, tensor_type=None):
        targets = torch.full((batch_size, self.out_size), fill_val,
                             requires_grad=False)
        if tensor_type is not None:
            targets = targets.to(tensor_type)
        return targets

    def disc_loss(self, in_data, labels):
        loss = self.loss_func(in_data, labels)
        return loss

    def forward(self, in_data):
        hidden_data = self.net(in_data)
        out_data = self.out_layer(hidden_data)
        return out_data
