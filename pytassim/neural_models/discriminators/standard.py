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
from pytassim.neural_models.linear_net import create_net


logger = logging.getLogger(__name__)


class StandardDisc(torch.nn.Module):
    """
    Standard discriminator for generative adversarial networks. This
    discriminator takes input data and returns processed data based on specified
    neural network. This discriminator can be also used as base discriminator to
    implement other types of discriminators for generative adversarial networks.

    Parameters
    ----------
    net : :py:class:`pytorch.nn.Module`
        This network is used for this discriminator.
    """
    def __init__(self, net,):
        super().__init__()
        self.net = net
        self.loss_func = torch.nn.BCEWithLogitsLoss()

    @staticmethod
    def get_targets(batch_size, fill_val=1.0, tensor_type=None):
        targets = torch.full((batch_size, 1), fill_val,
                             requires_grad=False)
        if tensor_type is not None:
            targets = targets.to(tensor_type)
        return targets

    def disc_loss(self, in_data, labels):
        loss = self.loss_func(in_data, labels)
        return loss

    def forward(self, in_data):
        out_data = self.net(in_data)
        return out_data
