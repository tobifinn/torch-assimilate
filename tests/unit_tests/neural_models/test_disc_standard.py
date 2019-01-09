#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 1/9/19

Created for torch-assimilate

@author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de

    Copyright (C) {2019}  {Tobias Sebastian Finn}

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# System modules
import unittest
import logging
import os

# External modules
import xarray as xr
import torch

# Internal modules
from pytassim.neural_models.discriminators.standard import StandardDisc


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestDiscStandard(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        net = torch.nn.Linear(16, 1)
        self.disc = StandardDisc(net)

    def test_loss_function_is_bce_with_logits(self):
        self.assertIsInstance(self.disc.loss_func, torch.nn.BCEWithLogitsLoss)

    def test_get_targets_returns_target_tensor(self):
        batch_size = 128
        target = torch.ones((batch_size, 1))
        returned_target = self.disc.get_targets(batch_size)
        torch.testing.assert_allclose(returned_target, target)

    def test_get_targets_uses_fill_value(self):
        batch_size = 128
        target = torch.full((batch_size, 1), 5)
        returned_target = self.disc.get_targets(batch_size, fill_val=5)
        torch.testing.assert_allclose(returned_target, target)

    def test_get_targets_sets_required_grad_to_false(self):
        batch_size = 128
        returned_target = self.disc.get_targets(batch_size)
        self.assertFalse(returned_target.requires_grad)

    def test_get_targets_sets_to_given_type(self):
        batch_size = 128
        target = torch.ones((batch_size, 1)).int()
        returned_target = self.disc.get_targets(batch_size, tensor_type=target)
        self.assertEqual(returned_target.type(), target.type())

    def test_get_targets_sets_to_givendtype(self):
        batch_size = 128
        target = torch.ones((batch_size, 1)).int()
        returned_target = self.disc.get_targets(batch_size, tensor_type=torch.int)
        self.assertEqual(returned_target.type(), target.type())

    def test_disc_loss_returns_loss(self):
        batch_size = 128
        target = torch.ones((batch_size, 1))
        in_data = torch.empty((batch_size, 1)).uniform_(0, 1)
        loss = self.disc.loss_func(in_data, target)
        returned_loss = self.disc.disc_loss(in_data, target)
        torch.testing.assert_allclose(returned_loss, loss)

    def test_forward_uses_net_and_outlayer(self):
        batch_size = 128
        in_data = torch.empty((batch_size, 16)).normal_(0, 1)
        out_data = self.disc.net(in_data)
        returned_data = self.disc.forward(in_data)
        torch.testing.assert_allclose(returned_data, out_data)


if __name__ == '__main__':
    unittest.main()
