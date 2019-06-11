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
from unittest.mock import MagicMock, call, patch

# External modules
import xarray as xr
import torch
import numpy as np

# Internal modules
from pytassim.toolbox.discriminator.standard import StandardDisc


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestDiscStandard(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        net = torch.nn.Linear(40, 1).double()
        self.disc = StandardDisc(net)

    @property
    def obs_torch(self):
        obs_vals = self.obs['observations'].values
        obs_torch = torch.from_numpy(obs_vals).double()
        return obs_torch

    def inject_missing(self):
        self.disc.loss_func = torch.nn.BCEWithLogitsLoss()
        self.disc.optimizer = torch.optim.SGD(
            self.disc.trainable_params, lr=0.1
        )

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
        in_data = torch.empty((batch_size, 40)).normal_(0, 1).double()
        out_data = self.disc.net(in_data)
        returned_data = self.disc.forward(in_data)
        torch.testing.assert_allclose(returned_data, out_data)

    def test_trainable_params_returns_net_params(self):
        network_params = self.disc.net.parameters()
        self.assertListEqual(list(network_params), self.disc.trainable_params)

    def test_trainable_params_skips_if_grad_not_needed(self):
        self.disc.net.bias.requires_grad = False
        network_params = [p for p in self.disc.net.parameters()
                          if p.requires_grad]
        self.assertListEqual(network_params, self.disc.trainable_params)

    def test_check_trainable_checks_if_loss_func_is_callable(self):
        self.disc.loss_func = '123'
        with self.assertRaises(TypeError):
            self.disc.check_trainable()

    def test_check_trainable_checks_if_optimizer_set(self):
        self.disc.loss_func = torch.nn.BCEWithLogitsLoss()
        with self.assertRaises(TypeError):
            self.disc.check_trainable()

    def test_check_trainable_checks_if_trainable_params(self):
        self.disc.loss_func = torch.nn.BCEWithLogitsLoss()
        self.disc.optimizer = torch.optim.SGD(self.disc.trainable_params,
                                              lr=0.1)
        for param in self.disc.net.parameters():
            param.requires_grad = False
        with self.assertRaises(ValueError):
            self.disc.check_trainable()

    def test_train_sets_network_to_train(self):
        self.inject_missing()
        self.disc.net.train = MagicMock()
        _ = self.disc.train(self.obs_torch, self.obs_torch)
        self.disc.net.train.assert_called_once()

    def test_train_checks_if_trainable(self):
        self.inject_missing()
        self.disc.check_trainable = MagicMock()
        _ = self.disc.train(self.obs_torch, self.obs_torch)
        self.disc.check_trainable.assert_called_once()

    def test_train_sets_optimizer_zero_grad(self):
        self.inject_missing()
        self.disc.optimizer.zero_grad = MagicMock()
        _ = self.disc.train(self.obs_torch, self.obs_torch)
        self.disc.optimizer.zero_grad.assert_called_once()

    def test_train_uses_forward_for_real_data(self):
        self.inject_missing()
        fake = self.obs_torch + 1
        real_return = self.disc.forward(self.obs_torch)
        fake_return = self.disc.forward(fake)
        self.disc.forward = MagicMock(side_effect=[real_return, fake_return])
        _ = self.disc.train(real_data=self.obs_torch, fake_data=fake, test=123)
        self.assertEqual(len(self.disc.forward.call_args_list[0][0]), 1)
        torch.testing.assert_allclose(self.disc.forward.call_args_list[0][0][0],
                                      self.obs_torch)
        self.assertEqual(len(self.disc.forward.call_args_list[0][1]), 1)
        self.assertDictEqual(self.disc.forward.call_args_list[0][1],
                             {'test': 123})

    def test_train_uses_disc_loss_for_real_loss(self):
        self.inject_missing()
        fake = self.obs_torch + 1
        real_return = self.disc.forward(self.obs_torch)
        fake_return = self.disc.forward(fake)
        real_labels = self.disc.get_targets(3, 1.0, real_return)
        fake_labels = self.disc.get_targets(3, 0.0, real_return)
        self.disc.forward = MagicMock(side_effect=[real_return, fake_return])
        self.disc.get_targets = MagicMock(
            side_effect=[real_labels, fake_labels]
        )
        self.disc.disc_loss = MagicMock(
            return_value=MagicMock(spec_set=torch.zeros((1, )).to(real_return))
        )
        _ = self.disc.train(real_data=self.obs_torch, fake_data=fake, test=123)

        self.assertEqual(len(self.disc.disc_loss.call_args_list[0][0]), 2)
        torch.testing.assert_allclose(
            self.disc.disc_loss.call_args_list[0][0][0], real_return
        )
        torch.testing.assert_allclose(
            self.disc.disc_loss.call_args_list[0][0][1], real_labels
        )

    def test_train_uses_forward_for_fake_data(self):
        self.inject_missing()
        fake = self.obs_torch + 1
        real_return = self.disc.forward(self.obs_torch)
        fake_return = self.disc.forward(fake)
        self.disc.forward = MagicMock(side_effect=[real_return, fake_return])
        _ = self.disc.train(real_data=self.obs_torch, fake_data=fake, test=123)
        self.assertEqual(len(self.disc.forward.call_args_list[1][0]), 1)
        torch.testing.assert_allclose(self.disc.forward.call_args_list[1][0][0],
                                      fake)
        self.assertEqual(len(self.disc.forward.call_args_list[1][1]), 1)
        self.assertDictEqual(self.disc.forward.call_args_list[1][1],
                             {'test': 123})

    def test_train_uses_disc_loss_for_fake_loss(self):
        self.inject_missing()
        fake = self.obs_torch + 1
        real_return = self.disc.forward(self.obs_torch)
        fake_return = self.disc.forward(fake)
        real_labels = self.disc.get_targets(3, 1.0, real_return)
        fake_labels = self.disc.get_targets(3, 0.0, real_return)
        self.disc.forward = MagicMock(side_effect=[real_return, fake_return])
        self.disc.get_targets = MagicMock(
            side_effect=[real_labels, fake_labels]
        )
        self.disc.disc_loss = MagicMock(
            return_value=MagicMock(spec_set=torch.zeros((1, )).to(real_return))
        )
        _ = self.disc.train(real_data=self.obs_torch, fake_data=fake, test=123)

        self.assertEqual(len(self.disc.disc_loss.call_args_list[1][0]), 2)
        torch.testing.assert_allclose(
            self.disc.disc_loss.call_args_list[1][0][0], fake_return
        )
        torch.testing.assert_allclose(
            self.disc.disc_loss.call_args_list[1][0][1], fake_labels
        )

    def test_train_backwards_total_loss(self):
        self.inject_missing()
        fake_data = self.obs_torch + 1
        real_loss = MagicMock(spec_set=torch.zeros((1, )).to(self.obs_torch))
        fake_loss = MagicMock(spec_set=torch.zeros((1, )).to(self.obs_torch))
        total_loss = MagicMock(spec_set=torch.zeros((1, )).to(self.obs_torch))
        fake_loss.backward = MagicMock()
        real_loss.backward = MagicMock()
        total_loss.backward = MagicMock()
        trg = 'pytassim.toolbox.discriminator.standard.StandardDisc.' \
              '_get_train_losses'
        with patch(trg, return_value=(total_loss, real_loss, fake_loss)):
            _ = self.disc.train(real_data=self.obs_torch, fake_data=fake_data)
        total_loss.backward.assert_called_once()
        fake_loss.backward.assert_not_called()
        real_loss.backward.assert_not_called()

    def test_train_calls_optimizer_step(self):
        self.inject_missing()
        fake_data = self.obs_torch + 1
        self.disc.optimizer.step = MagicMock()
        _ = self.disc.train(real_data=self.obs_torch, fake_data=fake_data)
        self.disc.optimizer.step.assert_called_once()

    def test_returns_losses(self):
        self.inject_missing()
        fake_data = self.obs_torch + 1

        real_loss = MagicMock(spec_set=torch.ones((1, 1)).to(self.obs_torch))
        fake_loss = MagicMock(spec_set=torch.ones((1, 1)).to(self.obs_torch))
        total_loss = real_loss + fake_loss
        self.disc.disc_loss = MagicMock(side_effect=[real_loss, fake_loss])

        returned_losses = self.disc.train(real_data=self.obs_torch,
                                          fake_data=fake_data)
        torch.testing.assert_allclose(returned_losses[0], total_loss)
        self.assertEqual(returned_losses[1], real_loss)
        self.assertEqual(returned_losses[2], fake_loss)

    def test_eval_sets_net_to_eval(self):
        self.inject_missing()
        fake_data = self.obs_torch + 1
        self.disc.net.eval = MagicMock()
        _ = self.disc.eval(self.obs_torch, fake_data)
        self.disc.net.eval.assert_called_once()

    def test_eval_uses_forward_for_real_data(self):
        self.inject_missing()
        fake = self.obs_torch + 1
        real_return = self.disc.forward(self.obs_torch)
        fake_return = self.disc.forward(fake)
        self.disc.forward = MagicMock(side_effect=[real_return, fake_return])
        _ = self.disc.eval(real_data=self.obs_torch, fake_data=fake, test=123)
        self.assertEqual(len(self.disc.forward.call_args_list[0][0]), 1)
        torch.testing.assert_allclose(self.disc.forward.call_args_list[0][0][0],
                                      self.obs_torch)
        self.assertEqual(len(self.disc.forward.call_args_list[0][1]), 1)
        self.assertDictEqual(self.disc.forward.call_args_list[0][1],
                             {'test': 123})

    def test_eval_uses_disc_loss_for_real_loss(self):
        self.inject_missing()
        fake = self.obs_torch + 1
        real_return = self.disc.forward(self.obs_torch)
        fake_return = self.disc.forward(fake)
        real_labels = self.disc.get_targets(3, 1.0, real_return)
        fake_labels = self.disc.get_targets(3, 0.0, real_return)
        self.disc.forward = MagicMock(side_effect=[real_return, fake_return])
        self.disc.get_targets = MagicMock(
            side_effect=[real_labels, fake_labels]
        )
        self.disc.disc_loss = MagicMock(
            return_value=MagicMock(spec_set=torch.zeros((1, )).to(real_return))
        )
        _ = self.disc.eval(real_data=self.obs_torch, fake_data=fake, test=123)

        self.assertEqual(len(self.disc.disc_loss.call_args_list[0][0]), 2)
        torch.testing.assert_allclose(
            self.disc.disc_loss.call_args_list[0][0][0], real_return
        )
        torch.testing.assert_allclose(
            self.disc.disc_loss.call_args_list[0][0][1], real_labels
        )

    def test_eval_uses_forward_for_fake_data(self):
        self.inject_missing()
        fake = self.obs_torch + 1
        real_return = self.disc.forward(self.obs_torch)
        fake_return = self.disc.forward(fake)
        self.disc.forward = MagicMock(side_effect=[real_return, fake_return])
        _ = self.disc.eval(real_data=self.obs_torch, fake_data=fake, test=123)
        self.assertEqual(len(self.disc.forward.call_args_list[1][0]), 1)
        torch.testing.assert_allclose(self.disc.forward.call_args_list[1][0][0],
                                      fake)
        self.assertEqual(len(self.disc.forward.call_args_list[1][1]), 1)
        self.assertDictEqual(self.disc.forward.call_args_list[1][1],
                             {'test': 123})

    def test_eval_uses_disc_loss_for_fake_loss(self):
        self.inject_missing()
        fake = self.obs_torch + 1
        real_return = self.disc.forward(self.obs_torch)
        fake_return = self.disc.forward(fake)
        real_labels = self.disc.get_targets(3, 1.0, real_return)
        fake_labels = self.disc.get_targets(3, 0.0, real_return)
        self.disc.forward = MagicMock(side_effect=[real_return, fake_return])
        self.disc.get_targets = MagicMock(
            side_effect=[real_labels, fake_labels]
        )
        self.disc.disc_loss = MagicMock(
            return_value=MagicMock(spec_set=torch.zeros((1, )).to(real_return))
        )
        _ = self.disc.eval(real_data=self.obs_torch, fake_data=fake, test=123)

        self.assertEqual(len(self.disc.disc_loss.call_args_list[1][0]), 2)
        torch.testing.assert_allclose(
            self.disc.disc_loss.call_args_list[1][0][0], fake_return
        )
        torch.testing.assert_allclose(
            self.disc.disc_loss.call_args_list[1][0][1], fake_labels
        )

    def test_eval_returns_losses(self):
        self.inject_missing()
        fake_data = self.obs_torch + 1

        real_loss = MagicMock(spec_set=torch.ones((1, 1)).to(self.obs_torch))
        fake_loss = MagicMock(spec_set=torch.ones((1, 1)).to(self.obs_torch))
        total_loss = real_loss + fake_loss
        self.disc.disc_loss = MagicMock(side_effect=[real_loss, fake_loss])

        returned_losses = self.disc.eval(real_data=self.obs_torch,
                                         fake_data=fake_data)
        torch.testing.assert_allclose(returned_losses[0], total_loss)
        self.assertEqual(returned_losses[1], real_loss)
        self.assertEqual(returned_losses[2], fake_loss)

    def test_gen_loss_uses_forward_for_fake_data(self):
        self.inject_missing()
        fake_return = self.disc.forward(self.obs_torch)
        self.disc.forward = MagicMock(return_value=fake_return)
        _ = self.disc.gen_loss(fake_data=self.obs_torch, test=123)
        self.assertEqual(len(self.disc.forward.call_args_list[0][0]), 1)
        torch.testing.assert_allclose(self.disc.forward.call_args_list[0][0][0],
                                      self.obs_torch)
        self.assertEqual(len(self.disc.forward.call_args_list[0][1]), 1)
        self.assertDictEqual(self.disc.forward.call_args_list[0][1],
                             {'test': 123})

    def test_gen_loss_uses_disc_loss_for_gen_loss(self):
        self.inject_missing()
        fake_return = self.disc.forward(self.obs_torch)
        real_labels = self.disc.get_targets(3, 1.0, fake_return)
        self.disc.forward = MagicMock(return_value=fake_return)
        self.disc.get_targets = MagicMock(return_value=real_labels)
        self.disc.disc_loss = MagicMock(
            return_value=MagicMock(spec_set=torch.zeros((1, )).to(fake_return))
        )
        _ = self.disc.gen_loss(fake_data=self.obs_torch, test=123)

        self.assertEqual(len(self.disc.disc_loss.call_args_list[0][0]), 2)
        torch.testing.assert_allclose(
            self.disc.disc_loss.call_args_list[0][0][0], fake_return
        )
        torch.testing.assert_allclose(
            self.disc.disc_loss.call_args_list[0][0][1], real_labels
        )

    def test_gen_loss_returns_gen_loss(self):
        self.inject_missing()
        self.disc.disc_loss = MagicMock(return_value=self.obs_torch)
        gen_loss = self.disc.gen_loss(self.obs_torch)
        torch.testing.assert_allclose(gen_loss, self.obs_torch)

    def test_recon_loss_uses_gen_loss(self):
        self.inject_missing()
        self.disc.gen_loss = MagicMock(return_value=self.obs_torch)
        recon_loss = self.disc.recon_loss(self.obs_torch, test=123,)
        self.assertEqual(len(self.disc.gen_loss.call_args_list[0][0]), 1)
        torch.testing.assert_allclose(
            self.disc.gen_loss.call_args_list[0][0][0], self.obs_torch
        )
        self.assertEqual(len(self.disc.gen_loss.call_args_list[0][1]), 1)
        self.assertDictEqual(self.disc.gen_loss.call_args_list[0][1],
                             {'test': 123})
        torch.testing.assert_allclose(recon_loss, self.obs_torch)

    def test_back_loss_uses_gen_loss(self):
        self.inject_missing()
        self.disc.gen_loss = MagicMock(return_value=self.obs_torch)
        back_loss = self.disc.back_loss(self.obs_torch, test=123,)
        self.assertEqual(len(self.disc.gen_loss.call_args_list[0][0]), 1)
        torch.testing.assert_allclose(
            self.disc.gen_loss.call_args_list[0][0][0], self.obs_torch
        )
        self.assertEqual(len(self.disc.gen_loss.call_args_list[0][1]), 1)
        self.assertDictEqual(self.disc.gen_loss.call_args_list[0][1],
                             {'test': 123})
        torch.testing.assert_allclose(back_loss, self.obs_torch)


if __name__ == '__main__':
    unittest.main()
