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

from xarray.backends.api import DATAARRAY_VARIABLE

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
        save_patch.assert_called_once_with('test.nc')

    def test_saves_dataset_passes_args_kwargs(self):
        with patch('xarray.core.dataset.Dataset.to_netcdf') as save_patch:
            save_netcdf(
                self.dataset, 'test.nc', 'r', format='nc4'

            )
        save_patch.assert_called_once_with(
            'test.nc', 'r', format='nc4'
        )

    def test_converts_array_to_dataset(self):
        with patch('xarray.DataArray.to_netcdf') as save_patch:
            save_netcdf(dataset_to_save=self.dataarray, save_path='test.nc')
        save_patch.assert_called_once_with('test.nc')

    def test_multidimensional_coordinates_are_transformed(self):
        multi_dim_ds = self.dataset.assign_coords(
            obs_grid_1=pd.MultiIndex.from_product(
                (np.arange(40), [0,]), names=['grid_point', 'height']
            )
        )
        single_dim_ds = self.dataset
        with patch('xarray.Dataset.to_netcdf') as save_patch, \
                patch(
                    'pytassim.utilities.xarray.encode_multidim',
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
        returned_dataset = encode_multidim(self.dataset)
        self.assertNotIsInstance(
            returned_dataset.indexes['obs_grid_1'], pd.MultiIndex
        )

    def test_dataset_resets_variables_to_coordinates(self):
        returned_dataset = encode_multidim(self.dataset)
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
        returned_dataset = encode_multidim(self.dataset)
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
        encoded_dataset = encode_multidim(self.dataset)
        encoded_dataset.to_netcdf('/tmp/test.nc')
        write_patch.assert_called_once()
        if os.path.isfile('/tmp/test.nc'):
            os.remove('/tmp/test.nc')


class TestLoadNC(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.dataarray = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.dataset = xr.open_dataset(obs_path).load()

    def test_opens_dataset(self):
        with patch(
                'xarray.open_dataset',
                return_value=self.dataset
        ) as open_patch:
            returned_dataset = load_netcdf('test.nc')

        open_patch.assert_called_once_with('test.nc')
        self.assertEqual(returned_dataset, self.dataset)

    def test_passes_args_kwargs_to_open_dataset(self):
        with patch(
                'xarray.open_dataset',
                return_value=self.dataset
        ) as open_patch:
            _ = load_netcdf(
                'test.nc', False, None, decode_cf=True
            )
        open_patch.assert_called_once_with('test.nc', None, decode_cf=True)

    def test_open_dataset_returns_array_with_array_keyword(self):
        with patch(
                'xarray.open_dataarray',
                return_value=self.dataarray
        ) as open_patch:
            returnd_array = load_netcdf(
                'test.nc', array=True, decode_cf=True
            )
        open_patch.assert_called_once_with('test.nc', decode_cf=True)
        xr.testing.assert_identical(returnd_array, self.dataarray)

    def test_decodes_multidimensional_coords(self):
        multi_dim_ds = self.dataset.assign_coords(
            obs_grid_1=pd.MultiIndex.from_product(
                (np.arange(40), [0,]), names=['grid_point', 'height']
            )
        )
        with patch(
                'xarray.open_dataset',
                return_value=multi_dim_ds
        ), patch(
            'pytassim.utilities.xarray.decode_multidim',
            return_value=self.dataset
        ) as decode_patch:
            _ = load_netcdf(
                'test.nc', decode_cf=True
            )
        decode_patch.assert_called_once_with(dataset=multi_dim_ds)


class TestDecodeMultidim(unittest.TestCase):
    def setUp(self):
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        dataset = xr.open_dataset(obs_path).load()
        self.dataset = dataset.assign_coords(
            obs_grid_1=pd.MultiIndex.from_product(
                (np.arange(40), [0,]), names=['grid_point', 'height']
            )
        )

    def test_cycling_of_encode_decode_get_same_dataset(self):
        single_dim_dataset = encode_multidim(self.dataset)
        returned_dataset = decode_multidim(single_dim_dataset)
        xr.testing.assert_identical(returned_dataset, self.dataset)

    def test_decode_decodes_multiindex(self):
        single_dim_dataset = encode_multidim(self.dataset)
        returned_dataset = decode_multidim(single_dim_dataset)
        self.assertIsInstance(returned_dataset.indexes['obs_grid_1'],
                              pd.MultiIndex)
        pd.testing.assert_index_equal(
            self.dataset.indexes['obs_grid_1'],
            returned_dataset.indexes['obs_grid_1']
        )

    def test_removes_attrs_from_dataset(self):
        single_dim_dataset = encode_multidim(self.dataset)
        returned_dataset = decode_multidim(single_dim_dataset)
        self.assertDictEqual(returned_dataset['obs_grid_1'].attrs, {})

    def test_removes_levels_from_coords(self):
        single_dim_dataset = encode_multidim(self.dataset)
        returned_dataset = decode_multidim(single_dim_dataset)
        self.assertNotIn('grid_point', returned_dataset.coords.keys())
        self.assertNotIn('height', returned_dataset.coords.keys())


if __name__ == '__main__':
    unittest.main()
