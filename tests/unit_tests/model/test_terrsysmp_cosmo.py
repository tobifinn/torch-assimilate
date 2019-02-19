#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2/19/19

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

# Internal modules
from pytassim.model.terrsysmp import cosmo


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = '/scratch/local1/Data/phd_thesis/test_data'


@unittest.skipIf(
    not os.path.isdir(DATA_PATH), 'Data for TerrSysMP not available!'
)
class TestTerrSysMPCosmo(unittest.TestCase):
    def setUp(self):
        self.dataset = xr.open_dataset(
            os.path.join(DATA_PATH, 'lffd20150731060000.nc')
        )
        self.assim_vars = ['T', 'W', 'W_SO', 'T_2M']

    def test_cycling_returns_identical_datasets(self):
        reconstructed_ds = cosmo.postprocess_cosmo(
            cosmo.preprocess_cosmo(self.dataset, self.assim_vars), self.dataset
        )
        xr.testing.assert_allclose(reconstructed_ds, self.dataset)

    def test_preprocess_selects_only_available_vars(self):
        ens_arr = cosmo.preprocess_cosmo(self.dataset, self.assim_vars+['TEMP'])
        self.assertListEqual(list(ens_arr.var_name), self.assim_vars)

    def test_preprocess_raises_logger_warning_for_filtered_vars(self):
        with self.assertLogs(level=logging.WARNING) as log:
            _ = cosmo.preprocess_cosmo(
                self.dataset, self.assim_vars+['TEMP']
            )
        with self.assertRaises(AssertionError):
            with self.assertLogs(level=logging.WARNING):
                _ = cosmo.preprocess_cosmo(
                    self.dataset, self.assim_vars
                )

    def test_prepare_vgrid_sets_levels(self):
        vcoord = self.dataset['vcoord']
        ds = self.dataset[self.assim_vars]
        ret_ds = cosmo._prepare_vgrid(ds, vcoord)
        right_soil1 = ds['soil1'].values*(-1)
        right_level1 = vcoord.values
        right_level = (vcoord.values+np.roll(vcoord.values, 1))[1:] / 2
        np.testing.assert_equal(ret_ds['soil1'].values, right_soil1)
        np.testing.assert_equal(ret_ds['level1'].values, right_level1)
        np.testing.assert_equal(ret_ds['level'].values, right_level)

    def test_prepare_vgrid_sets_vcoord_if_no_soil(self):
        self.assim_vars.remove('W_SO')
        vcoord = self.dataset['vcoord']
        ds = self.dataset[['T', 'W']]
        ret_ds = cosmo._prepare_vgrid(ds, vcoord)
        np.testing.assert_equal(ret_ds['vgrid'], vcoord.values)

    def test_prepare_vgrid_sets_concatenated_if_soil(self):
        vcoord = self.dataset['vcoord']
        ds = self.dataset[self.assim_vars]
        ret_ds = cosmo._prepare_vgrid(ds, vcoord)
        soil1 = self.dataset['soil1'] * (-1)
        right_vgrid = np.concatenate([vcoord.values, soil1.values], axis=0)
        np.testing.assert_equal(ret_ds['vgrid'], right_vgrid)

    def test_precosmo_calls_prepare_vgrid(self):
        ret_arr = cosmo.preprocess_cosmo(self.dataset, self.assim_vars)
        print(ret_arr)


if __name__ == '__main__':
    unittest.main()
