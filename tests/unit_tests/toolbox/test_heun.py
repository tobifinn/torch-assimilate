#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2/4/19

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
import torch

# Internal modules
from pytassim.toolbox.heun import HeunMethod

logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestHeunMethod(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(size=(8, 1))
        self.y = self.x * 4
        self.layer = torch.nn.Linear(1, 1, bias=False)
        self.loss = torch.nn.MSELoss()
        self.iters = 1
        self.optim = HeunMethod(self.layer.parameters(), lr=0.1)

    def closure(self):
        self.optim.zero_grad()
        y_pred = self.layer.forward(self.x)
        mse_loss = self.loss(y_pred, self.y)
        mse_loss.backward()
        return mse_loss

    def zero_closure(self):
        self.optim.zero_grad()
        y_pred = self.layer.forward(self.x)
        mse_loss = self.loss(y_pred, self.y)
        mse_loss.backward()
        self.optim.zero_grad()
        return mse_loss

    def test_step_optimize(self):
        old_params = [p.data for p in self.layer.parameters()][0].item()
        for i in range(self.iters):
            _ = self.closure()
            self.optim.step(closure=self.closure)
        new_params = [p.data for p in self.layer.parameters()][0].item()
        self.assertLess(abs(new_params - 4), abs(old_params-4))

    def test_step_sets_pred_grad_to_new_grad(self):
        for i in range(self.iters):
            _ = self.closure()
            self.optim.step(closure=self.zero_closure)
        param = list(self.layer.parameters())[0]
        self.assertEqual(self.optim.state[param]['pred_grad'].item(), 0)

    def test_fast_step_uses_pred_grad(self):
        _ = self.zero_closure()
        self.optim.step(closure=self.zero_closure)
        old_param = list(self.layer.parameters())[0].item()
        _ = self.closure()
        self.optim.step(closure=self.zero_closure)
        new_param = list(self.layer.parameters())[0].item()
        self.assertEqual(new_param, old_param)

    def test_step_uses_backward(self):
        self.optim.state['fast'] = False
        _ = self.zero_closure()
        self.optim.step(closure=self.zero_closure)
        old_param = list(self.layer.parameters())[0].item()
        _ = self.closure()
        self.optim.step(closure=self.zero_closure)
        new_param = list(self.layer.parameters())[0].item()
        self.assertNotEqual(new_param, old_param)
        self.assertLess(abs(new_param - 4), abs(old_param-4))

    def test_lr_raises_value_error_if_neg(self):
        with self.assertRaises(ValueError):
            self.optim = HeunMethod(self.layer.parameters(), lr=-0.1)


if __name__ == '__main__':
    unittest.main()
