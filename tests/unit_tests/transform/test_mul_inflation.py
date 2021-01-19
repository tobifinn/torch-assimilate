#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 19.01.21
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

# External modules
import xarray as xr
import numpy as np

# Internal modules
from pytassim.transform.mul_inflation import MultiplicativeInflation


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestMulInflation(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = (xr.open_dataset(obs_path).load(), )
        self.transform = MultiplicativeInflation(inf_factor=1.2)

    def test_inflate_array_inflates_array(self):
        state_mean = self.state.mean('ensemble')
        inflated_array = (self.state-state_mean) * np.sqrt(1.2) + state_mean
        ret_array = self.transform._inflate_array(self.state)
        xr.testing.assert_identical(ret_array, inflated_array)

    def test_inflate_array_inflates_cov_by_inf_factor(self):
        sliced_state = self.state[0, 0]
        state_mean = sliced_state.mean('ensemble')
        state_perts = sliced_state - state_mean
        state_perts_t = state_perts.rename({'grid': 'grid_new'})
        state_cov = xr.dot(
            state_perts, state_perts_t, dims='ensemble'
        )
        inflated_cov = state_cov * 1.2

        ret_array = self.transform._inflate_array(self.state)
        ret_sliced_state = ret_array[0, 0]
        ret_state_mean = ret_sliced_state.mean('ensemble')
        ret_state_perts = ret_sliced_state - ret_state_mean
        ret_state_perts_t = ret_state_perts.rename({'grid': 'grid_new'})
        ret_cov = xr.dot(
            ret_state_perts, ret_state_perts_t, dims='ensemble'
        )
        np.testing.assert_allclose(ret_cov.values, inflated_cov.values)

    def test_pre_inflates_background_and_first_guess(self):
        inflated_state = self.transform._inflate_array(self.state)
        ret_background, ret_obs, ret_fg = self.transform.pre(
            self.state, self.obs, self.state
        )
        xr.testing.assert_identical(ret_background, inflated_state)
        xr.testing.assert_identical(ret_fg, inflated_state)
        self.assertTupleEqual(ret_obs, self.obs)

    def test_pre_inflates_background_with_no_first_guess(self):
        inflated_state = self.transform._inflate_array(self.state)
        ret_background, ret_obs, ret_fg = self.transform.pre(
            self.state, self.obs, None
        )
        xr.testing.assert_identical(ret_background, inflated_state)
        self.assertIsNone(ret_fg)
        self.assertTupleEqual(ret_obs, self.obs)

    def test_post_inflates_analysis(self):
        inflated_state = self.transform._inflate_array(self.state)
        ret_analysis = self.transform.post(
            analysis=self.state,
            background=self.state+1,
            observations=self.obs,
            first_guess=self.state+1
        )
        xr.testing.assert_identical(ret_analysis, inflated_state)


if __name__ == '__main__':
    unittest.main()
