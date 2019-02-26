#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2/21/19

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
import xarray as xr
import numpy as np

# Internal modules
from pytassim.model.terrsysmp import clm


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = '/scratch/local1/Data/phd_thesis/test_data'


@unittest.skipIf(
    not os.path.isdir(DATA_PATH), 'Data for TerrSysMP not available!'
)
class TestCLMInterface(unittest.TestCase):
    def setUp(self):
        self.dataset = xr.open_dataset(
            os.path.join(DATA_PATH, 'clmoas_det.clm2.h0.2015-07-31-22500.nc')
        ).load()
        self.assim_vars = ['H2OSOI', 'TLAKE', 'WIND']
        self.dataset['TLAKE'] = self.dataset['TLAKE'].copy(
            data=np.ones_like(self.dataset['TLAKE'])
        )

    def test_dataset_can_be_reconstructed(self):
        analysis_data = clm.preprocess_clm(self.dataset, self.assim_vars)
        analysis_ds = clm.postprocess_clm(analysis_data, self.dataset)
        xr.testing.assert_identical(analysis_ds, self.dataset)


if __name__ == '__main__':
    unittest.main()
