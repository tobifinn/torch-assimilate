#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 3/19/18

Created for tf-assimilate

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
import xarray as xr

# Internal modules
from tfassim.observation import Observation


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


rnd = np.random.RandomState(42)


class TestObsSubset(unittest.TestCase):
    def setUp(self):
        self.covariance = np.ones(shape=(10, 10))
        self.obs = rnd.normal(size=(1, 10,))
        self.obs_nr = np.arange(self.obs.shape[-1])
        self.obs_time = np.arange(1)
        self.obs_ds = xr.Dataset(
            data_vars={
                'observations': (('time', 'obs_grid_1'), self.obs),
                'covariance': (
                    ('obs_grid_1', 'obs_grid_2'), self.covariance
                )
            },
            coords={
                'time': self.obs_time,
                'obs_grid_1': self.obs_nr,
                'obs_grid_2': self.obs_nr
            }
        )

    def test_xr_dataset_has_accessor(self):
        self.assertTrue(hasattr(self.obs_ds, 'obs'))


if __name__ == '__main__':
    unittest.main()
