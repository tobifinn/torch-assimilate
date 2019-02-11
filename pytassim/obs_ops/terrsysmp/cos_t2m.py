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
    def __init__(self, station_df, cosmo_coords, cosmo_height,
                 lev_inds=(40, 35)):
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
        lev_inds : tuple, optional
            These level indices are used to estimate the lapse rate.
        """
        self._h_diff = None
        self._locs = None
        self.station_df = station_df
        self.cosmo_coords = cosmo_coords
        self.cosmo_height = cosmo_height
        self.lev_inds = lev_inds

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
        cosmo_stacked = self.cosmo_height.stack(grid=['rlat', 'rlon'])
        cosmo_surf = cosmo_stacked.isel(level1=-1, time=0)
        cosmo_height = cosmo_surf.values[locs]
        height_diff = station_height - cosmo_height
        return height_diff

    @staticmethod
    def _get_neighbors(src_points, trg_points):
        tree = cKDTree(src_points)
        _, locs = tree.query(trg_points, k=1)
        return locs

    def get_lapse_rate(self, cosmo_ds):
        pass

    def obs_op(self, in_array, *args, **kwargs):
        pass
