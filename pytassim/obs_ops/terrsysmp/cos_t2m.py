#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 2/11/19
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2019}  {Tobias Sebastian Finn}
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# System modules
import logging

# External modules
from scipy.spatial import cKDTree
import numpy as np

# Internal modules
from ..base_ops import BaseOperator


logger = logging.getLogger(__name__)


EARTH_RADIUS = 6371000


class CosmoT2mOperator(BaseOperator):
    def __init__(self, station_df, cosmo_coords, cosmo_height,):
        """
        This 2-metre-temperature observation operator is used as observation
        operator for COSMO data. This observation operator selects the nearest
        grid point to given observations and corrects the height difference
        between COSMO and station height as written in the user guide of COSMO.

        Parameters
        ----------
        station_df : :py:class:`pandas.DataFrame`
            This station dataframe is used to determine the station position and
            height.
        cosmo_coords : :py:class:`numpy.ndarray`
            The cosmo coordinates as numpy array. The first axis should be the
            number of grid points, while the second axis is ('lat', 'lon').
        cosmo_height : :py:class:`xarray.DataArray`
            The constant cosmo height. This can be found in constant cosmo files
            as HHL.
        """
        self._h_diff = None
        self._locs = None
        self.station_df = station_df
        self.cosmo_coords = cosmo_coords
        self.cosmo_height = cosmo_height
        self.lev_inds = [35, 25]

    @property
    def locs(self):
        if self._locs is None:
            self._locs = self._calc_locs()
        return self._locs

    @property
    def height_diff(self):
        if self._h_diff is None:
            self._h_diff = self._calc_h_diff(self.locs)
        return self._h_diff

    @staticmethod
    def _get_cartesian(latlonalt):
        lat_rad = np.deg2rad(latlonalt[:, 0])
        lon_rad = np.deg2rad(latlonalt[:, 1])
        x = EARTH_RADIUS * np.cos(lat_rad) * np.cos(lon_rad)
        y = EARTH_RADIUS * np.cos(lat_rad) * np.sin(lon_rad)
        z = EARTH_RADIUS * np.sin(lat_rad) + latlonalt[:, 2]
        xyz = np.stack([x, y, z], axis=-1)
        return xyz

    def _calc_locs(self):
        station_lat_lon = self.station_df[['Breite', 'Länge']].values
        station_alt = self.station_df['Stations-\r\nhöhe'].values.reshape(-1, 1)
        station_llalt = np.concatenate([station_lat_lon, station_alt], axis=-1)
        station_xyz = self._get_cartesian(station_llalt)

        cosmo_alt = self.cosmo_height.isel(level1=-1, time=0).values
        cosmo_alt = cosmo_alt.reshape(-1, 1)
        cosmo_llalt = np.concatenate([self.cosmo_coords, cosmo_alt], axis=-1)
        cosmo_xyz = self._get_cartesian(cosmo_llalt)

        locs = self._get_neighbors(cosmo_xyz, station_xyz)
        return locs

    def _calc_h_diff(self, locs):
        station_height = self.station_df['Stations-\r\nhöhe'].values
        cosmo_loc = self._localize_grid(self.cosmo_height)
        cosmo_height = cosmo_loc.isel(level1=-1, time=0).values
        height_diff = station_height - cosmo_height
        return height_diff

    @staticmethod
    def _get_neighbors(src_points, trg_points):
        tree = cKDTree(src_points)
        _, locs = tree.query(trg_points, k=1)
        return locs

    def _localize_grid(self, ds):
        stacked_ds = ds.stack(grid=['rlat', 'rlon'])
        localized_ds = stacked_ds.isel(grid=self.locs)
        return localized_ds

    def _get_lapse_rate(self, cosmo_ds):
        h_full = self.cosmo_height - self.cosmo_height.isel(level1=-1)
        h_full = h_full.isel(level1=slice(None, -1))
        h_loc = self._localize_grid(h_full).isel(time=0)
        h_diff = h_loc.isel(level1=self.lev_inds[1]) - \
                 h_loc.isel(level1=self.lev_inds[0])

        temp_loc = self._localize_grid(cosmo_ds['T'])
        temp_diff = temp_loc.isel(level=self.lev_inds[1]) - \
                    temp_loc.isel(level=self.lev_inds[0])

        lapse_rate = temp_diff / h_diff
        return lapse_rate

    def obs_op(self, in_array, *args, **kwargs):
        uncorr_t2m = self._localize_grid(in_array['T_2M'])
        correction = self.height_diff * self._get_lapse_rate(in_array)
        corr_t2m = uncorr_t2m.squeeze(dim='height_2m') + correction
        return corr_t2m
