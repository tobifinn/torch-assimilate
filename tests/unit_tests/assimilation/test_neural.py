#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12/11/18

Created for torch-assimilate

@author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de

    Copyright (C) {2018}  {Tobias Sebastian Finn}

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
import torch
import xarray as xr
import numpy as np

# Internal modules
from pytassim.assimilation.neural.neural import NeuralAssimilation
from pytassim.testing import DummyNeuralModule, if_gpu_decorator


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestNeuralAssimilation(unittest.TestCase):
    def setUp(self):
        self.module = DummyNeuralModule()
        self.algorithm = NeuralAssimilation(model=self.module)
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()

    def test_dummy_module_if_nn_module(self):
        self.assertIsInstance(self.module, torch.nn.Module)

    def test_dummy_module_assimilate_returns_state(self):
        ret_state = self.module.assimilate(self.state, self.obs)
        xr.testing.assert_identical(ret_state, self.state)
        self.assertEqual(id(self.state), id(ret_state))

    def test_update_state_calls_prepare_obs(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        prepared_obs = self.algorithm._prepare_obs(obs_tuple)
        with patch('pytassim.assimilation.neural.neural.NeuralAssimilation.'
                   '_prepare_obs', return_value=prepared_obs) as prepare_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple, ana_time)
        prepare_patch.assert_called_once_with(obs_tuple)

    def test_update_state_transfers_tensors_to_torch(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        obs_state, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        torch_states = self.algorithm._states_to_torch(
            self.state.values, obs_state, obs_cov
        )
        trg = 'pytassim.assimilation.neural.neural.NeuralAssimilation.' \
              '_states_to_torch'
        with patch(trg, return_value=torch_states) as transfer_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple, ana_time)
        transfer_patch.assert_called_once()
        np.testing.assert_equal(transfer_patch.call_args[0][0],
                                self.state.values)
        np.testing.assert_equal(transfer_patch.call_args[0][1], obs_state)
        np.testing.assert_equal(transfer_patch.call_args[0][2], obs_cov)

    def test_update_state_calls_assimilate_from_module(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        obs_state, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        torch_states = self.algorithm._states_to_torch(
            self.state.values, obs_state, obs_cov
        )
        trg = 'pytassim.testing.dummy.DummyNeuralModule.assimilate'
        with patch(trg, return_value=torch_states[0]) as module_patch:
            _ = self.algorithm.update_state(self.state, obs_tuple, ana_time)
        module_patch.assert_called_once()
        torch.testing.assert_allclose(module_patch.call_args[0][0],
                                      torch_states[0])
        torch.testing.assert_allclose(module_patch.call_args[0][1],
                                      torch_states[1])
        torch.testing.assert_allclose(module_patch.call_args[0][2],
                                      torch_states[2])

    def test_update_state_returns_state_with_new_data(self):
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        obs_state, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        torch_states = self.algorithm._states_to_torch(
            self.state.values, obs_state, obs_cov
        )
        torch_states[0][0] = 9999
        analysis = self.state.copy(deep=True, data=torch_states[0].numpy())
        trg = 'pytassim.testing.dummy.DummyNeuralModule.assimilate'
        with patch(trg, return_value=torch_states[0]) as _:
            ret_ana = self.algorithm.update_state(self.state, obs_tuple,
                                                  ana_time)
        xr.testing.assert_identical(ret_ana, analysis)

    @if_gpu_decorator
    def test_update_state_copies_analysis_back_to_cpu(self):
        self.module = self.module.cuda()
        self.algorithm.gpu = True
        ana_time = self.state.time[-1].values
        obs_tuple = (self.obs, self.obs)
        obs_state, obs_cov, _ = self.algorithm._prepare_obs(obs_tuple)
        torch_states = self.algorithm._states_to_torch(
            self.state.values, obs_state, obs_cov
        )
        torch_states[0][0] = 9999
        analysis = self.state.copy(deep=True,
                                   data=torch_states[0].cpu().numpy())
        trg = 'pytassim.testing.dummy.DummyNeuralModule.assimilate'
        with patch(trg, return_value=torch_states[0]) as _:
            ret_ana = self.algorithm.update_state(self.state, obs_tuple,
                                                  ana_time)
        xr.testing.assert_identical(ret_ana, analysis)

    def test_module_gets_private_module(self):
        self.algorithm._model = 123
        self.assertEqual(self.algorithm.model, 123)

    def test_module_sets_private_module(self):
        self.algorithm._model = None
        self.algorithm.model = self.module
        self.assertIsInstance(self.algorithm._model, torch.nn.Module)

    def test_module_raises_type_error_if_not_assimilate(self):
        with self.assertRaises(TypeError):
            self.algorithm.model = self.state

    @if_gpu_decorator
    def test_module_copies_module_to_cpu_if_gpu_false(self):
        self.module.linear = torch.nn.Linear(16, 8)
        self.module = self.module.cuda()
        self.algorithm.gpu = False
        self.algorithm.model = self.module
        self.assertFalse(next(self.algorithm.model.parameters()).is_cuda)

    @if_gpu_decorator
    def test_module_copies_module_to_gpu_if_gpu_true(self):
        self.module.linear = torch.nn.Linear(16, 8)
        self.module = self.module.cpu()
        self.algorithm.gpu = True
        self.algorithm.model = self.module
        self.assertTrue(next(self.algorithm.model.parameters()).is_cuda)

    def test_module_casts_to_dtype(self):
        self.module.linear = torch.nn.Linear(16, 8)
        self.module = self.module.type(torch.float16)
        self.algorithm.model = self.module
        self.assertIsInstance(next(self.algorithm.model.parameters()),
                              torch.DoubleTensor)

    def test_model_copies_weights_to_new_model(self):
        self.module.linear = torch.nn.Linear(16, 8, bias=False)
        self.module = self.module.type(self.algorithm.dtype)
        parameter = torch.nn.Parameter(
            torch.ones_like(self.module.linear.weight)
        )
        self.module.linear.weight = parameter
        torch.testing.assert_allclose(self.module.linear.weight, parameter)
        self.algorithm.model = self.module
        torch.testing.assert_allclose(
            self.algorithm.model.linear.weight, parameter
        )


if __name__ == '__main__':
    unittest.main()
