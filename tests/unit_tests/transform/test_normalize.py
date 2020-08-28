#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12/18/18

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
import inspect

# External modules
import xarray as xr

# Internal modules
from pytassim.transform.normalize import Normalizer
from pytassim.testing.dummy import dummy_obs_operator


logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestNormalizer(unittest.TestCase):
    def setUp(self):
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        self.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        self.obs = xr.open_dataset(obs_path).load()
        self.norm = Normalizer((1, 2), ((1, 5), ), (1, 2))

    def test_pre_normalize_background(self):
        old_state = self.state.copy(deep=True)
        normalized_array = (self.state - 1) / 2
        obs_tuple = (self.obs, )
        bg_normed, _, _ = self.norm.pre(self.state, obs_tuple, self.state)
        xr.testing.assert_equal(bg_normed, normalized_array)
        xr.testing.assert_equal(self.state, old_state)

    def test_pre_normalize_first_guess(self):
        old_state = self.state.copy(deep=True)
        normalized_array = (self.state - 1) / 2
        obs_tuple = (self.obs, )
        _, _, fg_normed = self.norm.pre(self.state, obs_tuple, self.state)
        xr.testing.assert_equal(fg_normed, normalized_array)
        xr.testing.assert_equal(self.state, old_state)

    def test_pre_normalize_observations(self):
        old_obs = self.obs.copy()
        normalized_obs = self.obs.copy()
        normalized_obs['observations'] = (self.obs['observations'] - 1) / 5
        obs_tuple = (self.obs, )
        _, obs_normed, _ = self.norm.pre(self.state, obs_tuple, self.state)
        xr.testing.assert_equal(obs_normed[0], normalized_obs)
        xr.testing.assert_equal(self.obs, old_obs)

    def test_pre_normalize_many_obs(self):
        self.norm.obs_stat = ((1, 5), (1, 2))
        normalized_obs = [self.obs.copy(deep=True), self.obs.copy(deep=True)]
        normalized_obs[0]['observations'] = (normalized_obs[0]['observations']-1) / 5
        normalized_obs[1]['observations'] = (normalized_obs[1]['observations']-1) / 2
        obs_tuple = (self.obs, self.obs)
        _, obs_normed, _ = self.norm.pre(self.state, obs_tuple, self.state)
        xr.testing.assert_equal(
            obs_normed[0]['observations'], normalized_obs[0]['observations']
        )
        xr.testing.assert_equal(
            obs_normed[1]['observations'], normalized_obs[1]['observations']
        )

    def test_pre_normalize_sets_old_obs_operator(self):
        self.obs.obs.operator = dummy_obs_operator
        _, obs_normed, _ = self.norm.pre(self.state, (self.obs, ), self.state)
        right_ret_value = dummy_obs_operator(self.obs, self.state)
        ret_value = obs_normed[0].obs.operator(self.obs, self.state)
        xr.testing.assert_identical(ret_value, right_ret_value)

    def test_post_normalize_analysis_based_on_ens(self):
        old_state = self.state.copy(deep=True)
        normalized_array = self.state * 2 + 1
        obs_tuple = (self.obs, )
        analysis_normed = self.norm.post(
            self.state, self.state, obs_tuple, self.state
        )
        xr.testing.assert_equal(analysis_normed, normalized_array)
        xr.testing.assert_equal(self.state, old_state)


if __name__ == '__main__':
    unittest.main()
