#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 02.12.18
#
# Created for torch-assim
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
import os
import unittest
import logging

# External modules
import torch
import numpy as np

# Internal modules
from pytassim.model.lorenz_96 import torch_roll, Lorenz96


rnd = np.random.RandomState(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)


class TestLorenz96(unittest.TestCase):
    def setUp(self):
        self.model = Lorenz96()
        self.state = rnd.normal(size=40)
        self.torch_state = torch.tensor(self.state)

    def test_torch_roll(self):
        rolled_array = np.roll(self.state, shift=4, axis=0)
        returned_array = torch_roll(self.torch_state, 4).numpy().copy()
        np.testing.assert_equal(returned_array, rolled_array)
        rolled_array = np.roll(self.state, shift=-4, axis=0)
        returned_array = torch_roll(self.torch_state, shift=-4).numpy().copy()
        np.testing.assert_equal(returned_array, rolled_array)

    def test_torch_roll_axis(self):
        state = rnd.normal(size=(20, 40, 20))
        torch_state = torch.tensor(state)
        rolled_array = np.roll(state, shift=3, axis=-1)
        returned_array = torch_roll(torch_state, shift=3, axis=-1).numpy()
        np.testing.assert_equal(returned_array, rolled_array)
        rolled_array = np.roll(state, shift=-4, axis=1)
        returned_array = torch_roll(torch_state, shift=-4, axis=1).numpy()
        np.testing.assert_equal(returned_array, rolled_array)

    def test_calc_forcing_returns_forcing(self):
        returned_forcing = self.model._calc_forcing(self.torch_state)
        self.assertEqual(returned_forcing, self.model.forcing)
        self.model.forcing = 2
        returned_forcing = self.model._calc_forcing(self.torch_state)
        self.assertEqual(returned_forcing, self.model.forcing)

    def test_calc_advection_returns_advection_term(self):
        diff = np.roll(self.state, -1) - np.roll(self.state, 2)
        right_advection = diff * np.roll(self.state, 1)
        returned_advection = self.model._calc_advection(self.torch_state)
        np.testing.assert_equal(right_advection, returned_advection.numpy())

    def test_calc_advection_rolls_along_last_axis(self):
        state = rnd.normal(size=(20, 40))
        torch_state = torch.tensor(state)
        diff = np.roll(state, -1, axis=-1) - np.roll(state, 2, axis=-1)
        right_advection = diff * np.roll(state, 1, axis=-1)
        returned_advection = self.model._calc_advection(torch_state)
        np.testing.assert_equal(right_advection, returned_advection)

    def test_calc_dissipation_returns_dissipation_term(self):
        dissipation = -self.state
        returned_dissipation = self.model._calc_dissipation(self.torch_state)
        np.testing.assert_equal(dissipation, returned_dissipation.numpy())

    def test_call_returns_state_update(self):
        state_update = self.model._calc_advection(self.torch_state)
        state_update += self.model._calc_dissipation(self.torch_state)
        state_update += self.model._calc_forcing(self.torch_state)
        returned_update = self.model(self.torch_state)
        torch.testing.assert_allclose(state_update, returned_update)


if __name__ == '__main__':
    unittest.main()
