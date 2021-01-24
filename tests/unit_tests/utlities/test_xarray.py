#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 24.01.21
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import unittest
import logging
import os
from mock import patch

# External modules
import xarray as xr
import numpy as np
import pandas as pd

# Internal modules
from pytassim.utilities.xarray import *


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestSaveNC(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.dataarray = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.dataset = xr.open_dataset(obs_path).load()

    def test_saves_dataset_file_to_path(self):
        with patch('xarray.core.dataset.Dataset.to_netcdf') as save_patch:
            save_netcdf(dataset_to_save=self.dataset, save_path='test.nc')
        save_patch.assert_called_once_with(path='test.nc')

    def test_converts_array_to_dataset(self):
        state_dataset = self.dataarray.to_dataset(
            name='__xarray_dataarray_variable__'
        )
        with patch('xarray.Dataset.to_netcdf') as save_patch, \
                patch(
                    'xarray.DataArray.to_dataset', return_value=state_dataset
                ) as dataset_patch:
            save_netcdf(dataset_to_save=self.dataarray, save_path='test.nc')

        dataset_patch.assert_called_once_with(
            name='__xarray_dataarray_variable__'
        )
        save_patch.assert_called_once_with(path='test.nc')

    def test_multidimensional_coordinates_are_transformed(self):
        multi_dim_ds = self.dataset.assign_coords(
            obs_grid_1=pd.MultiIndex.from_product(
                (np.arange(40), [0,]), names=['grid_point', 'height']
            )
        )
        single_dim_ds = self.dataset
        with patch('xarray.Dataset.to_netcdf') as save_patch, \
                patch(
                    'pytassim.utilities.xarray.encode_multidim_dataset',
                    return_value=single_dim_ds
                ) as dim_patch:
            save_netcdf(dataset_to_save=multi_dim_ds, save_path='test.nc')
        dim_patch.assert_called_once_with(dataset=multi_dim_ds)


class TestMultiToSingleDim(unittest.TestCase):
    def setUp(self):
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        dataset = xr.open_dataset(obs_path).load()
        self.dataset = dataset.assign_coords(
            obs_grid_1=pd.MultiIndex.from_product(
                (np.arange(40), [0,]), names=['grid_point', 'height']
            )
        )

    def test_dataset_has_single_indexes_only(self):
        returned_dataset = encode_multidim_dataset(self.dataset)
        self.assertNotIsInstance(
            returned_dataset.indexes['obs_grid_1'], pd.MultiIndex
        )

    def test_dataset_resets_variables_to_coordinates(self):
        returned_dataset = encode_multidim_dataset(self.dataset)
        self.assertNotIn('grid_point', returned_dataset.dims)
        self.assertNotIn('height', returned_dataset.dims)
        self.assertIn('grid_point', returned_dataset.coords.keys())
        self.assertIn('height', returned_dataset.coords.keys())
        self.assertTupleEqual(
            returned_dataset['grid_point'].dims, ('obs_grid_1',)
        )
        self.assertTupleEqual(
            returned_dataset['height'].dims, ('obs_grid_1',)
        )
        np.testing.assert_equal(
            returned_dataset['grid_point'].values, np.arange(40)
        )
        np.testing.assert_equal(returned_dataset['height'].values, 0)

    def test_dataset_sets_attribute(self):
        returned_dataset = encode_multidim_dataset(self.dataset)
        self.assertIn(
            'multidim_levels', returned_dataset['obs_grid_1'].attrs.keys()
        )
        self.assertEqual(
            returned_dataset['obs_grid_1'].attrs['multidim_levels'],
            'grid_point;height'
        )

    @patch('xarray.backends.common.AbstractWritableDataStore.set_variables')
    def test_dataset_after_encoding_can_be_stored(self, write_patch):
        with self.assertRaises(NotImplementedError):
            self.dataset.to_netcdf('/tmp/test.nc')
        encoded_dataset = encode_multidim_dataset(self.dataset)
        encoded_dataset.to_netcdf('/tmp/test.nc')
        write_patch.assert_called_once()
        if os.path.isfile('/tmp/test.nc'):
            os.remove('/tmp/test.nc')


if __name__ == '__main__':
    unittest.main()
