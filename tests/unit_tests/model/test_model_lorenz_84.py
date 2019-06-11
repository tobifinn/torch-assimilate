#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12/4/18

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

# External modules
import numpy as np
import torch

# Internal modules
from pytassim.model.lorenz_84 import Lorenz84


rnd = np.random.RandomState(42)
logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestLorenz84(unittest.TestCase):
    def setUp(self):
        self.model = Lorenz84()
        self.state = rnd.normal(size=3)
        self.torch_state = torch.tensor(self.state)

    def test_current_returns_current_gradient(self):
        coupling = -self.state[..., 1] ** 2 - self.state[..., 2] ** 2
        damping = self.model.damp_factor * self.state[..., 0]
        forcing = self.model.damp_factor * self.model.symm_forcing
        right_grad = coupling - damping + forcing
        returned_grad = self.model._calc_westerly(self.torch_state)
        np.testing.assert_equal(right_grad, returned_grad.numpy())

    def test_calc_cosine_phase_returns_cosine_gradient(self):
        amp = self.state[..., 0] * self.state[..., 1]
        displace = -self.model.dis_factor * self.state[..., 0] * \
                   self.state[..., 2]
        damping = self.state[..., 1]
        forcing = self.model.asymm_forcing
        right_grad = amp + displace - damping + forcing
        returned_grad = self.model._calc_cosine_phase(self.torch_state)
        np.testing.assert_equal(right_grad, returned_grad.numpy())

    def test_calc_sine_phase_returns_sine_gradient(self):
        amp = self.state[..., 0] * self.state[..., 2]
        displace = self.model.dis_factor * self.state[..., 0] * \
                   self.state[..., 1]
        damping = self.state[..., 2]
        right_grad = amp + displace - damping
        returned_grad = self.model._calc_sine_phase(self.torch_state)
        np.testing.assert_equal(right_grad, returned_grad.numpy())

    def test_call_returns_stacked_grad(self):
        westerly = self.model._calc_westerly(self.torch_state)
        cosine_phase = self.model._calc_cosine_phase(self.torch_state)
        sine_phase = self.model._calc_sine_phase(self.torch_state)
        stacked_arr = torch.stack([westerly, cosine_phase, sine_phase], dim=-1)
        returned_arr = self.model(self.torch_state)
        torch.testing.assert_allclose(stacked_arr, returned_arr)


if __name__ == '__main__':
    unittest.main()
