#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 1/11/19

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
from unittest.mock import MagicMock

# External modules
import torch

# Internal modules
from pytassim.toolbox.loss import LossWrapper


logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


class TestLossWrapper(unittest.TestCase):
    def setUp(self):
        self.torch_loss = torch.nn.MSELoss()
        self.loss_wrapper = LossWrapper(self.torch_loss)

    def test_loss_gets_private_loss(self):
        self.loss_wrapper._loss = 123
        self.assertEqual(self.loss_wrapper.loss, 123)

    def test_loss_sets_private_loss(self):
        self.loss_wrapper._loss = None
        self.loss_wrapper.loss = self.torch_loss
        self.assertEqual(id(self.torch_loss), id(self.loss_wrapper._loss))

    def test_loss_checks_if_callable(self):
        with self.assertRaises(TypeError):
            self.loss_wrapper.loss = 123

    def test_recon_loss_passes_recon_obs_and_obs_to_loss(self):
        self.loss_wrapper._loss = MagicMock(return_value=10)
        recon_loss = self.loss_wrapper.recon_loss(
            recon_obs=1, observation=2, prior=3, prior_ensemble=4, noise=5
        )
        self.assertEqual(recon_loss, 10)
        self.loss_wrapper._loss.assert_called_once_with(input=1, target=2)

    def test_back_loss_passes_analysis_and_prior_to_loss(self):
        self.loss_wrapper._loss = MagicMock(return_value=10)
        back_loss = self.loss_wrapper.back_loss(
            analysis=1, observation=2, prior=3, prior_ensemble=4, noise=5
        )
        self.assertEqual(back_loss, 10)
        self.loss_wrapper._loss.assert_called_once_with(input=1, target=3)

    def test_back_loss_passes_analysis_and_ens_if_no_prior(self):
        self.loss_wrapper._loss = MagicMock(return_value=10)
        back_loss = self.loss_wrapper.back_loss(
            analysis=1, observation=2, prior=None, prior_ensemble=4, noise=5
        )
        self.assertEqual(back_loss, 10)
        self.loss_wrapper._loss.assert_called_once_with(input=1, target=4)

    def test_back_loss_raises_argument_error_if_no_prior_given(self):
        self.loss_wrapper._loss = MagicMock(return_value=10)
        with self.assertRaises(ValueError):
            _ = self.loss_wrapper.back_loss(
                analysis=1, observation=2, prior=None, prior_ensemble=None,
                noise=5
            )


if __name__ == '__main__':
    unittest.main()
