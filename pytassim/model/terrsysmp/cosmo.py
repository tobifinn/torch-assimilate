#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 2/13/19
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
from typing import Iterable

# External modules
import numpy as np
import scipy.spatial.distance
import xarray as xr

# Internal modules
from . import common


logger = logging.getLogger(__name__)


__all__ = ['preprocess_cosmo', 'postprocess_cosmo']

_cosmo_vcoords = ['height_2m', 'height_10m', 'height_toa', 'soil1',
                  'level1', 'level', 'no_vgrid']


def preprocess_cosmo(
        cosmo_ds: xr.Dataset,
        assim_vars: Iterable[str]
) -> xr.DataArray:
    """
    This function can be used to pre-process COSMO data. There are different
    pre-processing steps included:

    1. Select variables to assimilate with `assim_vars`
    2. Set missing vertical levels of COSMO based on `vcoord`
    3. Interpolate all vertical levels of COSMO to a merge vertical grid
    4. Stack horizontal and vertical grid to one single dimension
    5. Transpose dimensions such that the resulting state array is valid

    Parameters
    ----------
    cosmo_ds : :py:class:`xarray.Dataset`
        From this COSMO dataset selected variables are extracted and this
        dataset is converted into a :py:class:`xarray.DataArray`. This dataset
        needs `vcoord` as data variable to determine the vertical coordinates of
        COSMO.
    assim_vars : iterable(str)
        These variables are included in the resulting and prepared array. If a
        variable cannot be found within the data, a warning will be raised.

    Returns
    -------
    prepared_data : :py:class:`xarray.DataArray`
        This array is prepared such that it could be assimilate by any
        assimilation algorithm within this package. The vertical grid is unified
        and stack together with the horizontal grid. The dataset with the
        selected variables is converted into this array, where 'var_name'
        indicates the variable axis.
    """
    avail_vars = [var for var in assim_vars if var in cosmo_ds.data_vars]
    not_avail_vars = list(set(assim_vars) - set(avail_vars))
    if not_avail_vars:
        logger.warning('Following variables are not found! {0:s}'.format(
            ', '.join(not_avail_vars)
        ))
    assim_ds = cosmo_ds[avail_vars]
    vgrid_ds = _prepare_vgrid(assim_ds, cosmo_ds['vcoord'])
    added_ds = common.add_no_vgrid(vgrid_ds, _cosmo_vcoords, 0)
    interp_ds = _interp_vgrid(added_ds)
    prepared_ds = _replace_coords(interp_ds)
    prepared_data = common.ds_to_array(prepared_ds, ('rlat', 'rlon', 'vgrid'))
    return prepared_data


def postprocess_cosmo(
        analysis_data: xr.DataArray,
        cosmo_ds: xr.Dataset
) -> xr.Dataset:
    """
    This function can be used to post-process COSMO data and incorporate
    included variables into given COSMO dataset. There are different steps
    included:

    1. Unstack the grid form analysis data and convert the data into a dataset
    2. Iterate through included variables and reindex these variables as they
        are in given COSMO dataset
    3. Replace the variables in given COSMO dataset with the analysis variables

    Parameters
    ----------
    analysis_data : :py:class:`xarray.DataArray`
        This array represents a valid analysis state, which can be a result of
        any given assimilation algorithm included in this package. In this
        analysis data, the horizontal and vertical grid are stacked together and
        the variables are a dimension within the array.
    cosmo_ds : :py:class:`xarray.Dataset`
        This COSMO dataset is used as source dataset to convert given analysis
        array into a valid COSMO dataset. The resulting dataset is a copy of
        this dataset, where the assimilated variables are replaced.

    Returns
    -------
    analysis_ds : :py:class:`xarray.Dataset`
        This analysis dataset is a copy of given COSMO dataset with replaced
        variables from given analysis array.
    """
    analysis_ds = common.generic_postprocess(
        analysis_data, cosmo_ds,
        vcoords=_cosmo_vcoords
    )
    logger.info('Finished post-processing of COSMO')
    return analysis_ds


def _prepare_vgrid(ds: xr.Dataset, vcoord: xr.DataArray) -> xr.Dataset:
    ds = ds.copy()
    dims_non_vert = [d for d in vcoord.dims if d not in _cosmo_vcoords]
    vcoord_vals = vcoord.mean(dim=dims_non_vert).values
    if 'soil1' in ds.coords:
        ds['soil1'] = ds['soil1'].copy(data=ds['soil1']*(-1))
        vgrid_coords = np.concatenate([vcoord_vals, ds['soil1'].values])
    else:
        vgrid_coords = vcoord_vals
    ds = ds.assign_coords(vgrid=vgrid_coords)
    if 'level1' in ds.dims:
        ds['level1'] = vcoord_vals
    if 'level' in ds.dims:
        ds['level'] = ((vcoord_vals+np.roll(vcoord_vals, 1))/2)[1:]
    return ds


def _interp_vgrid(ds: xr.Dataset) -> xr.Dataset:
    vgrid_neighbor_funcs = {
        'no_vgrid': _inds_nearest,
        'height_2m': _inds_nearest,
        'height_10m': _inds_nearest,
        'height_toa': _inds_nearest,
        'soil1': _inds_bottom,
        'level1': _inds_top,
        'level': _inds_top
    }
    vertical_coords = [c for c in _cosmo_vcoords if c in ds.coords]
    for c in vertical_coords:
        vgrid_inds = vgrid_neighbor_funcs[c](ds[c].values, ds['vgrid'].values)
        ds[c] = ds[c].copy(data=ds['vgrid'].values[vgrid_inds])
        ds = ds.reindex(**{c: ds['vgrid'].values}, method=None)
    return ds


def _inds_nearest(coord_val: np.ndarray, vgrid_val: np.ndarray) -> np.ndarray:
    dist_matrix = scipy.spatial.distance.cdist(
        coord_val[:, None], vgrid_val[:, None]
    )
    dist_argmin = np.argmin(dist_matrix, axis=1)
    return dist_argmin


def _inds_top(coord_val: np.ndarray, vgrid_val: np.ndarray) -> np.ndarray:
    return np.arange(len(vgrid_val))[:len(coord_val)]


def _inds_bottom(coord_val: np.ndarray, vgrid_val: np.ndarray) -> np.ndarray:
    return np.arange(len(vgrid_val))[-len(coord_val):]


def _replace_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_vertical = {c: 'vgrid' for c in _cosmo_vcoords}
    ds = common.replace_grid(ds, rename_vertical)
    rename_horizontal = {'srlat': 'rlat', 'srlon': 'rlon'}
    ds = common.replace_grid(ds, rename_horizontal)
    return ds
