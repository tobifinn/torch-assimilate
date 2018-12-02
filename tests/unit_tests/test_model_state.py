#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 3/23/18

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
import datetime as dt

# External modules
import numpy as np
import xarray as xr

# Internal modules
from pytassim.state import ModelState


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


rnd = np.random.RandomState(42)


class TestState(unittest.TestCase):
    def setUp(self):
        self.values = rnd.normal(size=(2, 1, 5, 100))
        self.dims = ('variable', 'time', 'ensemble', 'grid')
        self.state_da = xr.DataArray(
            data=self.values,
            coords={
                'variable': ['T', 'RH', ],
                'time': [dt.datetime(year=1992, month=12, day=25), ],
                'ensemble': np.arange(5),
                'grid': np.arange(100)
            },
            dims=self.dims
        )
    
    def test_array_has_state_accessor(self):
        self.assertTrue(hasattr(self.state_da, 'state'))

    def test_array_split_mean_perts_returns_two_dataarrays(self):
        mean, pert = self.state_da.state.split_mean_perts()
        self.assertIsInstance(mean, xr.DataArray)
        self.assertIsInstance(pert, xr.DataArray)

    def test_array_split_returns_mean_of_ensemble(self):
        right_mean = self.state_da.mean(dim='ensemble')
        returned_mean, _ = self.state_da.state.split_mean_perts()
        xr.testing.assert_identical(right_mean, returned_mean)

    def test_array_split_mean_uses_dim(self):
        for var in self.dims:
            right_mean = self.state_da.mean(dim=var)
            returned_mean, _ = self.state_da.state.split_mean_perts(var)
            xr.testing.assert_identical(right_mean, returned_mean)

    def test_array_split_mean_uses_axis(self):
        for k, _ in enumerate(self.dims):
            right_mean = self.state_da.mean(axis=k)
            returned_mean, _ = self.state_da.state.split_mean_perts(
                dim=None, axis=k
            )
            xr.testing.assert_identical(right_mean, returned_mean)

    def test_array_split_mean_raises_value_error_if_both_given(self):
        with self.assertRaises(ValueError):
            _ = self.state_da.state.split_mean_perts(dim='ensemble', axis=0)

    def test_array_split_mean_takes_sequence(self):
        right_mean = self.state_da.mean(dim=('ensemble', 'grid'))
        returned_mean, _ = self.state_da.state.split_mean_perts(
            dim=('ensemble', 'grid')
        )
        xr.testing.assert_identical(right_mean, returned_mean)

    def test_array_split_returns_perts_as_second(self):
        mean = self.state_da.mean('ensemble')
        right_perts = self.state_da - mean
        _, returned_perts = self.state_da.state.split_mean_perts()
        xr.testing.assert_identical(right_perts, returned_perts)

    def test_array_split_mean_pert_are_corresponding(self):
        right_mean = self.state_da.mean('ensemble')
        right_perts = self.state_da - right_mean
        mean, perts = self.state_da.state.split_mean_perts()
        xr.testing.assert_identical(right_mean, mean)
        xr.testing.assert_identical(right_perts, perts)

    def test_array_split_kwargs_are_passed_mean(self):
        self.state_da.attrs['unit'] = 'degC'
        right_mean = self.state_da.mean('ensemble', keep_attrs=True)
        returned_mean, _ = self.state_da.state.split_mean_perts(keep_attrs=True)
        xr.testing.assert_identical(right_mean, right_mean)
        self.assertDictEqual(right_mean.attrs, returned_mean.attrs)

    def test_array_split_kwargs_are_passed_perts(self):
        self.state_da.attrs['unit'] = 'degC'
        mean = self.state_da.mean('ensemble', keep_attrs=True)
        right_perts = self.state_da - mean
        _, returned_perts = self.state_da.state.split_mean_perts(
            keep_attrs=True
        )
        xr.testing.assert_identical(right_perts, returned_perts)
        self.assertDictEqual(right_perts.attrs, returned_perts.attrs)

    def test_valid_dims_checks_dim_names_order(self):
        self.assertTrue(self.state_da.state._valid_dims)
        state_da = self.state_da.rename({'ensemble': 'test'})
        self.assertFalse(state_da.state._valid_dims)
        state_da = self.state_da.transpose(
            'grid', 'ensemble', 'time', 'variable'
        )
        self.assertFalse(state_da.state._valid_dims)

    def test_valid_type_checks_dtype_of_coords(self):
        self.assertTrue(self.state_da.state._valid_coord_type)
        self.state_da['ensemble'] = ['det', 1, 2, 3, 4]
        self.assertFalse(self.state_da.state._valid_coord_type)

    def test_valid_checks_dim_names(self):
        self.assertTrue(self.state_da.state.valid)
        state_da = self.state_da.rename({'ensemble': 'test'})
        self.assertFalse(state_da.state.valid)

    def test_valid_checks_coord_dtype(self):
        self.assertTrue(self.state_da.state.valid)
        self.state_da['ensemble'] = ['det', 1, 2, 3, 4]
        self.assertFalse(self.state_da.state.valid)


if __name__ == '__main__':
    unittest.main()
