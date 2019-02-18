#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 1/30/19

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
from unittest.mock import patch

# External modules
import xarray as xr
import numpy as np
import torch

# Internal modules
from pytassim.obs_ops.lorenz_96.identity import IdentityOperator
from pytassim.obs_ops.lorenz_96.bernoulli import BernoulliOperator


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestBernoulliOps(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path)
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path)
        self.operator = BernoulliOperator()

    def test_np_sigmoid_returns_sigmoid_function(self):
        torch_state = torch.from_numpy(self.state.values)
        torch_sig = torch.nn.Sigmoid()(torch_state).numpy()
        ret_sig = self.operator._np_sigmoid(self.state)
        np.testing.assert_almost_equal(ret_sig, torch_sig)

    def test_obs_op_calls_identity_obs_op(self):
        ret_state = IdentityOperator.obs_op(self.operator, self.state)
        with patch('pytassim.obs_ops.lorenz_96.identity.IdentityOperator.'
                   'obs_op', return_value=ret_state) as identity_patch:
            _ = self.operator.obs_op(self.state, 1, test=2)
        identity_patch.assert_called_once_with(self.state, 1, test=2)

    def test_obs_op_shifts_and_sigmoid(self):
        self.operator.obs_points = None
        self.operator.shift = -2

        pseudo_obs = self.operator._np_sigmoid(self.state.sel(var_name='x') + 2)
        ret_obs = self.operator.obs_op(self.state)

        xr.testing.assert_identical(ret_obs, pseudo_obs)

    def test_torch_operator_returns_same_as_obs_op(self):
        self.operator.obs_points = [1, 2, 3]
        self.operator.shift = -55
        pseudo_obs = self.operator.obs_op(self.state)
        pseudo_obs = pseudo_obs.values.reshape(-1, 3)

        torch_op = self.operator.torch_operator()
        torch_state = torch.from_numpy(self.state.sel(var_name='x').values)
        torch_state = torch_state.view(-1, self.operator.len_grid).float()
        ret_obs = torch_op(torch_state).numpy()

        np.testing.assert_almost_equal(ret_obs, pseudo_obs)

    def test_torch_operator_parameter_no_req_gradient(self):
        torch_op = self.operator.torch_operator()
        for param in torch_op.parameters():
            self.assertFalse(param.requires_grad)


if __name__ == '__main__':
    unittest.main()
