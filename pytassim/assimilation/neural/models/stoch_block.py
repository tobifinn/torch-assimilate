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
from collections import OrderedDict

# External modules
import torch
import torch.nn.functional

# Internal modules


logger = logging.getLogger(__name__)


class SNLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, n_power_iterations=1, eps=1e-12):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.register_buffer(
            'weights_u', torch.nn.functional.normalize(torch.Tensor(out_features).normal_(), dim=0, eps=eps)
        )
        self.register_buffer(
            'weights_v', torch.nn.functional.normalize(torch.Tensor(in_features).normal_(), dim=0, eps=eps)
        )
        self.register_buffer(
            'weights_sigma', torch.nn.functional.normalize(torch.Tensor(1).zero_(), dim=0, eps=eps)
        )

    def reset_parameters(self):
        super().reset_parameters()

    def power_iter(self, u, weight_matrix):
        v = torch.nn.functional.normalize(torch.mv(weight_matrix.t(), u), dim=0, eps=self.eps)
        u = torch.nn.functional.normalize(torch.mv(weight_matrix, v), dim=0, eps=self.eps)
        return u, v

    def update_sigma(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        u = getattr(self, 'weights_u')
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                u, v = self.power_iter(u, w_mat)
            sigma = torch.dot(u, torch.mv(w_mat, v))
        setattr(self, 'weights_u', u)
        setattr(self, 'weights_v', v)
        setattr(self, 'weights_sigma', sigma)

    @property
    def w_norm(self):
        return self.weight / getattr(self, 'weights_sigma', 1.)

    def forward(self, input):
        self.update_sigma()
        return torch.nn.functional.linear(input, self.w_norm, self.bias)

    def extra_repr(self):
        return '{0:s}, SN=True'.format(super().extra_repr())


def linear_layer(in_size, out_size, spectral_norm=True, batch_norm=False,):
    if spectral_norm:
        funcs = [
            SNLinear(in_size, out_size, bias=True)
        ]
    elif batch_norm:
        funcs = [
            torch.nn.Linear(in_size, out_size, bias=False),
            torch.nn.BatchNorm1d(out_size),
        ]
    else:
        funcs = [
            torch.nn.Linear(in_size, out_size, bias=True)
        ]

    funcs.append(torch.nn.LeakyReLU())


    layer = torch.nn.Sequential(*funcs)
    return layer


def create_net(in_size, out_size, data_dims, spectral_norm=True,
               batch_norm=False):
    net = OrderedDict()
    curr_dim = in_size
    for k, dim in enumerate(data_dims):
        net['Lin_{0:d}'.format(k)] = linear_layer(
            curr_dim, dim, spectral_norm, batch_norm
        )
        curr_dim = dim
    net['Lin_out'] = linear_layer(
        curr_dim, out_size, spectral_norm, batch_norm
    )
    return torch.nn.Sequential(net)


class Discriminator(torch.nn.Module):
    def __init__(self, in_size, out_size=1, hidden_size=16, disc_dims=(64, ),
                 spectral_norm=True, batch_norm=False):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.disc_dims = disc_dims

        self.loss_func = torch.nn.BCEWithLogitsLoss()

        self.net = create_net(self.in_size, hidden_size, disc_dims,
                              spectral_norm, batch_norm)
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


class StochBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, hidden_size=64, data_dims=(64, ),
                 rand_dims=(64, ), spectral_norm=True, batch_norm=False):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.loss_func = torch.nn.MSELoss()

        self.data_net = create_net(self.in_size, hidden_size, data_dims,
                                   spectral_norm, batch_norm)
        self.rand_net = create_net(self.out_size[0], hidden_size, rand_dims,
                                   spectral_norm, batch_norm)
        self.out_data_layer = torch.nn.Linear(self.hidden_size,
                                              self.out_size[0])
        self.out_rand_layer = torch.nn.Linear(self.hidden_size,
                                              self.out_size[1])

    def gen_rand_data(self, batch_size, tensor_type=None):
        rand_vec = torch.normal(
            torch.zeros((batch_size, self.out_size[0])),
            torch.ones((batch_size, self.out_size[0])),
        )
        if tensor_type is not None:
            rand_vec = rand_vec.to(tensor_type)
        return rand_vec

    def recon_loss(self, reconstruction, original):
        loss = self.loss_func(reconstruction, original)
        return loss

    def forward(self, in_data, rand_data):
        hidden_data = self.data_net(in_data)
        hidden_rand = self.rand_net(rand_data)

        hidden_state = hidden_data * hidden_rand

        out_data = self.out_data_layer(hidden_state)
        out_rand = self.out_rand_layer(hidden_state)
        return out_data, out_rand
