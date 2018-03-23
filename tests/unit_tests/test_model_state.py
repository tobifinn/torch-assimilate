#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 3/23/18

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
import datetime as dt

# External modules
import numpy as np
import xarray as xr

# Internal modules
from tfassim.state import ModelState


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


rnd = np.random.RandomState(42)


class TestState(unittest.TestCase):
    def setUp(self):
        self.values = rnd.normal(size=(1, 1, 1, 100))
        self.dims = ('variable', 'time', 'ensemble', 'grid')
        self.state_da = xr.DataArray(
            data=self.values,
            coords={
                'variable': ['T', ],
                'time': [dt.datetime(year=1992, month=12, day=25), ],
                'ensemble': [0, ],
                'grid': np.arange(100)
            },
            dims=self.dims
        )
    
    def test_array_has_state_accessor(self):
        self.assertTrue(hasattr(self.state_da, 'state'))


if __name__ == '__main__':
    unittest.main()
