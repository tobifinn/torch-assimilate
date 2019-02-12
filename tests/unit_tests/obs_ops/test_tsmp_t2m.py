#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2/11/19

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
from unittest.mock import MagicMock

# External modules
import pandas as pd
import xarray as xr
import numpy as np

# Internal modules
from pytassim.obs_ops.terrsysmp.cos_t2m import CosmoT2mOperator, EARTH_RADIUS


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = '/scratch/local1/Data/phd_thesis/test_data'


@unittest.skipIf(
    not os.path.isdir(DATA_PATH), 'Data for TerrSysMP not available!'
)
class TestCOST2m(unittest.TestCase):
    def setUp(self):
        self.station_df = pd.read_hdf(
            os.path.join(DATA_PATH, 'avail_stations.hd5'),
            'stations'
        )
        self.hhl_file = xr.open_dataset(os.path.join(DATA_PATH, 'cos_hhl.nc'))
        self.cos_coords = np.loadtxt(os.path.join(DATA_PATH, 'cos_coords.txt'))
        self.obs_op = CosmoT2mOperator(self.station_df, self.cos_coords,
                                       self.hhl_file)

        self.ens_file = xr.open_dataset(os.path.join(DATA_PATH, 'cos_ens_0.nc'))

    def test_calc_locs_uses_station_coords(self):
        station_lat_lon = self.station_df[['Breite', 'Länge']].values
        station_alt = self.station_df['Stations-\r\nhöhe'].values.reshape(-1, 1)
        station_llalt = np.concatenate([station_lat_lon, station_alt], axis=-1)
        xyz = self.obs_op._get_cartesian(station_llalt)
        self.obs_op._get_cartesian = MagicMock(return_value=xyz)
        _ = self.obs_op._calc_locs()
        np.testing.assert_equal(
            self.obs_op._get_cartesian.call_args_list[0][0][0],
            station_llalt
        )

    def test_calc_locs_uses_cos_coords(self):
        cos_lat_lon = self.cos_coords
        cos_hhl = self.hhl_file['HSURF'].isel(time=0).values.reshape(-1, 1)
        cos_llalt = np.concatenate([cos_lat_lon, cos_hhl], axis=-1)
        xyz = self.obs_op._get_cartesian(cos_llalt)
        self.obs_op._get_cartesian = MagicMock(return_value=xyz)
        _ = self.obs_op._calc_locs()
        np.testing.assert_equal(
            self.obs_op._get_cartesian.call_args_list[1][0][0],
            cos_llalt
        )

    def test_get_cartesian_returns_cartesian_from_llalt(self):
        cos_lat_lon = self.cos_coords
        cos_hhl = self.hhl_file['HSURF'].isel(time=0).values.reshape(-1, 1)
        cos_llalt = np.concatenate([cos_lat_lon, cos_hhl], axis=-1)

        lat_rad = np.deg2rad(cos_lat_lon[:, 0])
        lon_rad = np.deg2rad(cos_lat_lon[:, 1])
        x = EARTH_RADIUS * np.cos(lat_rad) * np.cos(lon_rad)
        y = EARTH_RADIUS * np.cos(lat_rad) * np.sin(lon_rad)
        z = EARTH_RADIUS * np.sin(lat_rad) + cos_llalt[:, 2]
        xyz = np.stack([x, y, z], axis=-1)
        ret_xyz = self.obs_op._get_cartesian(cos_llalt)

        np.testing.assert_equal(ret_xyz, xyz)

    def test_calc_locs_calls_get_neighbors_with_cos_station(self):
        cos_lat_lon = self.cos_coords
        cos_hhl = self.hhl_file['HSURF'].isel(time=0).values.reshape(-1, 1)
        cos_llalt = np.concatenate([cos_lat_lon, cos_hhl], axis=-1)
        cos_xyz = self.obs_op._get_cartesian(cos_llalt)

        station_lat_lon = self.station_df[['Breite', 'Länge']].values
        station_alt = self.station_df['Stations-\r\nhöhe'].values.reshape(-1, 1)
        station_llalt = np.concatenate([station_lat_lon, station_alt], axis=-1)
        station_xyz = self.obs_op._get_cartesian(station_llalt)

        neighbors = self.obs_op._get_neighbors(cos_xyz, station_xyz)
        self.obs_op._get_neighbors = MagicMock(return_value=neighbors)

        _ = self.obs_op._calc_locs()
        self.obs_op._get_neighbors.assert_called_once()
        np.testing.assert_equal(self.obs_op._get_neighbors.call_args[0][0],
                                cos_xyz)
        np.testing.assert_equal(self.obs_op._get_neighbors.call_args[0][1],
                                station_xyz)

    def test_calc_locs_returns_locs(self):
        cos_lat_lon = self.cos_coords
        cos_hhl = self.hhl_file['HSURF'].isel(time=0).values.reshape(-1, 1)
        cos_llalt = np.concatenate([cos_lat_lon, cos_hhl], axis=-1)
        cos_xyz = self.obs_op._get_cartesian(cos_llalt)

        station_lat_lon = self.station_df[['Breite', 'Länge']].values
        station_alt = self.station_df['Stations-\r\nhöhe'].values.reshape(-1, 1)
        station_llalt = np.concatenate([station_lat_lon, station_alt], axis=-1)
        station_xyz = self.obs_op._get_cartesian(station_llalt)

        locs = self.obs_op._get_neighbors(cos_xyz, station_xyz)

        ret_locs = self.obs_op._calc_locs()
        np.testing.assert_equal(ret_locs, locs)

    def test_locs_returns_locs(self):
        self.obs_op._locs = 10
        self.assertEqual(self.obs_op.locs, 10)

    def test_locs_calls_calc_locs(self):
        self.obs_op._locs = None
        locs = self.obs_op._calc_locs()
        self.obs_op._calc_locs = MagicMock(return_value=locs)
        _ = self.obs_op.locs
        self.obs_op._calc_locs.assert_called_once_with()
        np.testing.assert_equal(self.obs_op._locs, locs)

    def test_calc_hdiff_returns_hdiff(self):
        station_height = self.station_df['Stations-\r\nhöhe'].values
        cos_stacked = self.hhl_file['HSURF'].stack(grid=['rlat', 'rlon'])
        cos_surf = cos_stacked.isel(time=0)
        cos_height = cos_surf.values[self.obs_op.locs]
        height_diff = station_height - cos_height

        ret_diff = self.obs_op._calc_h_diff()
        np.testing.assert_equal(ret_diff, height_diff)

    def test_height_diff_returns_private(self):
        self.obs_op._h_diff = 10
        self.assertEqual(self.obs_op.height_diff, 10)

    def test_height_diff_calls_calc(self):
        self.obs_op._h_diff = None
        h_diff = self.obs_op._calc_h_diff()
        self.obs_op._calc_h_diff = MagicMock(return_value=h_diff)
        _ = self.obs_op.height_diff
        self.obs_op._calc_h_diff.assert_called_once_with()
        np.testing.assert_equal(self.obs_op._h_diff, h_diff)

    def test_get_lapse_rate_returns_lapse_rate(self):
        time_axis = self.ens_file.time
        time_axis = xr.concat([time_axis, time_axis+1], dim='time')
        self.ens_file = self.ens_file.sel(time=time_axis, method='nearest')

        heights_full = self.hhl_file['HHL'] - \
                       self.hhl_file['HHL'].isel(level1=-1)
        heights_full = heights_full.isel(level1=slice(None, -1))
        heights_stacked = heights_full.stack(grid=['rlat', 'rlon'])
        heights_loc = heights_stacked.isel(grid=self.obs_op.locs, time=0)
        heights_diff = heights_loc.isel(level1=self.obs_op.lev_inds[1]) - \
                       heights_loc.isel(level1=self.obs_op.lev_inds[0])

        temp_stacked = self.ens_file['T'].stack(grid=['rlat', 'rlon'])
        temp_loc = temp_stacked.isel(grid=self.obs_op._locs)
        temp_diff = temp_loc.isel(level=self.obs_op.lev_inds[1]) - \
                    temp_loc.isel(level=self.obs_op.lev_inds[0])

        lapse_rate = temp_diff / heights_diff

        ret_lapse_rate = self.obs_op.get_lapse_rate(self.ens_file)
        xr.testing.assert_equal(ret_lapse_rate, lapse_rate)

    def test_obs_op_returns_t2m(self):
        time_axis = self.ens_file.time
        time_axis = xr.concat([time_axis, time_axis+1], dim='time')
        self.ens_file = self.ens_file.sel(time=time_axis, method='nearest')
        uncorr_t2m = self.obs_op._localize_grid(self.ens_file['T_2M'])
        uncorr_t2m = uncorr_t2m.squeeze(dim='height_2m')
        lapse_rate = self.obs_op.get_lapse_rate(self.ens_file)
        correction = self.obs_op.height_diff * lapse_rate
        corr_t2m = uncorr_t2m + correction

        ret_t2m = self.obs_op.obs_op(self.ens_file)
        xr.testing.assert_equal(corr_t2m, ret_t2m)


if __name__ == '__main__':
    unittest.main()
