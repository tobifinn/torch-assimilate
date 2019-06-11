#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 1/10/19

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
from unittest.mock import MagicMock, patch
import itertools


# External modules
import torch
import xarray as xr
import numpy as np

# Internal modules
from pytassim.toolbox.autoencoder import Autoencoder


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class InferenceNet(torch.nn.Module):    # pragma: no cover
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(80, 40).float()

    def forward(self, observation, prior=None, prior_ensemble=None, noise=None):
        return self.net(observation.float())


class TestAutoencoder(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        infer_net = InferenceNet()
        obs_operator = torch.nn.Linear(40, 40)
        self.autoencoder = Autoencoder(infer_net, obs_operator)

    @property
    def obs_torch(self):
        obs_vals = np.tile(self.obs['observations'].values, (1, 2))
        obs_torch = torch.from_numpy(obs_vals)
        return obs_torch

    def inject_missing(self):
        self.autoencoder.recon_loss = self.autoencoder.back_loss = MagicMock()
        self.autoencoder.recon_loss.recon_loss = MagicMock(
            return_value=MagicMock(side_set=torch.zeros((1,),).requires_grad_())
        )
        self.autoencoder.back_loss.back_loss = MagicMock(
            return_value=MagicMock(side_set=torch.zeros((1,),).requires_grad_())
        )
        self.autoencoder.optimizer = torch.optim.SGD(
            self.autoencoder.trainable_params, lr=0.1
        )

    def test_autoencoder_parameters_contains_infer_and_obs_op(self):
        infer_net = torch.nn.Linear(80, 40)
        obs_operator = torch.nn.Linear(10, 40)
        parameters = list(infer_net.parameters())
        parameters.extend(list(obs_operator.parameters()))

        autoencoder = Autoencoder(infer_net, obs_operator)
        self.assertListEqual(parameters, autoencoder.trainable_params)

    def test_autoencoder_parameters_skips_requires_grad(self):
        infer_net = torch.nn.Linear(80, 40)
        obs_operator = torch.nn.Linear(10, 40)
        for param in obs_operator.parameters():
            param.requires_grad = False
        parameters = list(infer_net.parameters())

        autoencoder = Autoencoder(infer_net, obs_operator)
        self.assertListEqual(parameters, autoencoder.trainable_params)

    def test_forward_calls_inference_net_with_parameters(self):
        params = dict(
            observation=1,
            prior=2,
            prior_ensemble=3,
            noise=4
        )
        self.autoencoder.inference_net = MagicMock(
            spec_set=torch.nn.Module(), return_value=10
        )
        self.autoencoder.obs_operator = MagicMock(
            spec_set=torch.nn.Module(), return_value=0
        )
        analysis, recon_obs = self.autoencoder.forward(**params)
        self.autoencoder.inference_net.assert_called_once_with(**params)
        self.autoencoder.obs_operator.assert_called_once_with(10)
        self.assertEqual(analysis, 10)
        self.assertEqual(recon_obs, 0)

    def test_trainable_raises_typeerror_if_wrong_reconloss(self):
        with self.assertRaises(TypeError):
            self.autoencoder.check_trainable()

    def test_trainable_raises_typeerror_if_wrong_backloss(self):
        self.autoencoder.recon_loss = MagicMock()
        self.autoencoder.recon_loss.recon_loss = MagicMock()
        with self.assertRaises(TypeError):
            self.autoencoder.check_trainable()

    def test_trainable_raises_typeerror_if_wrong_optimizer(self):
        self.autoencoder.back_loss = self.autoencoder.recon_loss = MagicMock()
        self.autoencoder.recon_loss.recon_loss = MagicMock()
        self.autoencoder.back_loss.back_loss = MagicMock()
        with self.assertRaises(TypeError):
            self.autoencoder.check_trainable()

    def test_trainable_raises_valueerror_if_empty_params(self):
        self.autoencoder.back_loss = self.autoencoder.recon_loss = MagicMock()
        self.autoencoder.recon_loss.recon_loss = MagicMock()
        self.autoencoder.back_loss.back_loss = MagicMock()
        self.autoencoder.optimizer = MagicMock(spec_set=torch.optim.RMSprop)

        self.autoencoder.inference_net = torch.nn.Sequential()
        self.autoencoder.obs_operator = torch.nn.Sequential()

        with self.assertRaises(ValueError):
            self.autoencoder.check_trainable()

    def test_train_calls_check_trainable(self):
        self.inject_missing()

        self.autoencoder.check_trainable = MagicMock(
            return_value=None
        )
        _ = self.autoencoder.train(self.obs_torch)
        self.autoencoder.check_trainable.assert_called_once()

    def test_train_sets_nets_to_train_mode(self):
        self.inject_missing()

        self.autoencoder.inference_net.train = MagicMock(return_value=None)
        self.autoencoder.obs_operator.train = MagicMock(return_value=None)
        self.autoencoder.train(self.obs_torch)
        self.autoencoder.inference_net.train.assert_called_once()
        self.autoencoder.obs_operator.train.assert_called_once()

    def test_train_sets_optimizer_zero_grad(self):
        self.inject_missing()
        self.autoencoder.optimizer.zero_grad = MagicMock(return_value=None)
        _ = self.autoencoder.train(self.obs_torch)
        self.autoencoder.optimizer.zero_grad.assert_called_once()

    def test_train_calls_forward_with_params(self):
        self.inject_missing()
        data_dict = dict(
            observation=self.obs_torch, prior=1, prior_ensemble=2, noise=3
        )
        forward_return = self.autoencoder.forward(**data_dict)
        self.autoencoder.forward = MagicMock(return_value=forward_return)
        _ = self.autoencoder.train(**data_dict)
        self.autoencoder.forward.assert_called_once_with(**data_dict)

    def test_train_calls_back_loss_with_analysis_params(self):
        self.inject_missing()
        data_dict = dict(
            observation=self.obs_torch, prior=1, prior_ensemble=2, noise=3
        )
        analysis, recon_obs = self.autoencoder.forward(**data_dict)
        self.autoencoder.forward = MagicMock(return_value=(analysis, recon_obs))
        _ = self.autoencoder.train(**data_dict)
        self.autoencoder.back_loss.back_loss.assert_called_once_with(
            analysis, **data_dict
        )

    def test_train_calls_recon_loss_with_recon_obs_params(self):
        self.inject_missing()
        data_dict = dict(
            observation=self.obs_torch, prior=1, prior_ensemble=2, noise=3
        )
        analysis, recon_obs = self.autoencoder.forward(**data_dict)
        self.autoencoder.forward = MagicMock(return_value=(analysis, recon_obs))
        _ = self.autoencoder.train(**data_dict)
        self.autoencoder.recon_loss.recon_loss.assert_called_once_with(
            recon_obs, **data_dict)

    def test_train_calls_total_loss_backward(self):
        self.inject_missing()
        data_dict = dict(
            observation=self.obs_torch, prior=1, prior_ensemble=2, noise=3
        )
        total_loss, back_loss, recon_loss = MagicMock(), MagicMock(), \
                                            MagicMock()
        total_loss.backward = MagicMock()
        back_loss.backward = MagicMock()
        recon_loss.backward = MagicMock()
        with patch(
            'pytassim.toolbox.autoencoder.Autoencoder._get_train_losses',
            return_value=(total_loss, back_loss, recon_loss)
        ) as loss_patch:
            recon_loss = self.autoencoder.recon_loss.recon_loss()
            recon_loss.backward = MagicMock()
            _ = self.autoencoder.train(**data_dict)
        total_loss.backward.assert_called_once()
        recon_loss.backward.assert_not_called()
        back_loss.backward.assert_not_called()

    def test_train_calls_optimizer_step(self):
        self.inject_missing()
        data_dict = dict(
            observation=self.obs_torch, prior=1, prior_ensemble=2, noise=3
        )
        self.autoencoder.optimizer.step = MagicMock()
        _ = self.autoencoder.train(**data_dict)
        self.autoencoder.optimizer.step.assert_called_once()

    def test_train_returns_losses(self):
        self.inject_missing()
        data_dict = dict(
            observation=self.obs_torch, prior=1, prior_ensemble=2, noise=3
        )
        back_loss = self.autoencoder.back_loss.back_loss()
        recon_loss = self.autoencoder.recon_loss.recon_loss()
        total_loss = back_loss + recon_loss

        returned_losses = self.autoencoder.train(**data_dict)
        torch.testing.assert_allclose(returned_losses[0], total_loss)
        torch.testing.assert_allclose(returned_losses[1], back_loss)
        torch.testing.assert_allclose(returned_losses[2], recon_loss)

    def test_eval_sets_nets_to_eval(self):
        self.inject_missing()
        self.autoencoder.inference_net.eval = MagicMock(return_value=None)
        self.autoencoder.obs_operator.eval = MagicMock(return_value=None)
        _ = self.autoencoder.eval(self.obs_torch)
        self.autoencoder.inference_net.eval.assert_called_once()
        self.autoencoder.obs_operator.eval.assert_called_once()

    def test_eval_uses_train_losses(self):
        self.inject_missing()
        self.autoencoder._get_train_losses = MagicMock(
            return_value=(1, 2, 3)
        )
        data_dict = dict(
            observation=self.obs_torch, prior=1, prior_ensemble=2, noise=3
        )
        returned_losses = self.autoencoder.eval(data_dict)
        self.assertTupleEqual(returned_losses, (1, 2, 3))
        self.autoencoder._get_train_losses.assert_called_once()


if __name__ == '__main__':
    unittest.main()
