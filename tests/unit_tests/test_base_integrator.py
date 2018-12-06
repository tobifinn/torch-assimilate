#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11/29/18

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
from unittest.mock import patch
import logging
import os

# External modules
import numpy as np

# Internal modules
from pytassim.model.integration.integrator import BaseIntegrator
from pytassim.testing import dummy_model


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


class TestBaseIntegrator(unittest.TestCase):
    def setUp(self):
        self.integrator = BaseIntegrator(model=dummy_model)

    def test_dt_gets_private_dt(self):
        self.integrator._dt = None
        self.assertIsNone(self.integrator.dt)
        self.integrator._dt = 10
        self.assertEqual(self.integrator.dt, 10)

    def test_dt_sets_private_dt(self):
        self.integrator._dt = None
        self.integrator.dt = 10
        self.assertEqual(self.integrator._dt, 10)

    def test_dt_checks_if_float_and_value(self):
        with self.assertRaises(TypeError):
            self.integrator.dt = 'bla'
        with self.assertRaises(ValueError):
            self.integrator.dt = 0

    def test_model_gets_private_model(self):
        self.integrator._model = None
        self.assertIsNone(self.integrator.model)
        self.integrator._model = 10
        self.assertEqual(self.integrator.model, 10)

    def test_model_sets_private_model(self):
        self.integrator._model = None
        self.integrator.model = dummy_model
        self.assertEqual(self.integrator._model, dummy_model)

    def test_model_checks_if_callable(self):
        with self.assertRaises(TypeError):
            self.integrator.model = 10
        with self.assertRaises(TypeError):
            self.integrator.model = 'test'
        with self.assertRaises(TypeError):
            self.integrator.model = BaseIntegrator(dummy_model)

    @patch('pytassim.model.integration.integrator.BaseIntegrator.'
           '_calc_increment', return_value=np.array([1]))
    def test_integrate_passes_state_to_calc_increment(self, inc_patch):
        curr_state = np.array([5, ])
        _ = self.integrator.integrate(curr_state)
        inc_patch.assert_called_once_with(curr_state)

    @patch('pytassim.model.integration.integrator.BaseIntegrator.'
           '_calc_increment', return_value=np.array([1]))
    def test_integrate_returns_updated_state(self, inc_patch):
        curr_state = np.array([5, ])
        returned_state = self.integrator.integrate(curr_state)
        np.testing.assert_equal(returned_state, np.array([6,]))


if __name__ == '__main__':
    unittest.main()
