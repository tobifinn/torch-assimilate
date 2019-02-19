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

# External modules
import xarray as xr
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


__all__ = ['preprocess_cosmo', 'postprocess_cosmo']

_cosmo_vcoords = ['height_2m', 'height_10m', 'height_toa', 'soil1',
                  'level1', 'level']


def postprocess_cosmo(analysis_data, cosmo_ds):
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
    unstacked_analysis = analysis_data.unstack('grid')
    transposed_analysis = unstacked_analysis.transpose(
        'var_name', 'ensemble', 'time', 'vgrid', 'rlat', 'rlon'
    )
    pre_analysis_ds = transposed_analysis.to_dataset(dim='var_name')
    analysis_ds = cosmo_ds.copy(deep=True)
    for var in pre_analysis_ds.data_vars:
        try:
            reindex_vcoord = _get_vcoord_ind(analysis_ds[var])
            if reindex_vcoord is None:
                reindexed_ana_var = pre_analysis_ds[var].isel(vgrid=0)
            else:
                reindexed_ana_var = pre_analysis_ds[var].reindex(
                    vgrid=reindex_vcoord.values, method='nearest'
                )
            analysis_ds[var] = analysis_ds[var].copy(
                data=reindexed_ana_var.values
            )
        except KeyError:
            logger.warning('Var: {0:s} is not found'.format(var))
    return analysis_ds


def preprocess_cosmo(cosmo_ds, assim_vars):
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
            ','.join(not_avail_vars)
        ))
    assim_ds = cosmo_ds[avail_vars]
    vgrid_ds = _prepare_vgrid(assim_ds, cosmo_ds['vcoord'])
    interp_ds = _interp_vgrid(vgrid_ds)
    unified_ds = _replace_coords(interp_ds)
    unified_data = unified_ds.to_array(dim='var_name')
    stacked_data = unified_data.stack(grid=('rlat', 'rlon', 'vgrid'))
    if 'ensemble' not in stacked_data.dims:
        stacked_data = stacked_data.expand_dims(dim='ensemble')
    prepared_data = stacked_data.transpose('var_name', 'time', 'ensemble',
                                           'grid')
    return prepared_data


def _prepare_vgrid(ds, vcoord):
    ds = ds.copy()
    vcoord_vals = vcoord.values.reshape(-1, vcoord.shape[-1])[0, :]
    if 'soil1' in ds.coords:
        ds['soil1'] *= -1
        vgrid_coords = np.concatenate([vcoord_vals, ds['soil1'].values])
    else:
        vgrid_coords = vcoord_vals
    ds = ds.assign_coords(vgrid=vgrid_coords)
    if 'level1' in ds.dims:
        ds['level1'] = vcoord_vals
    if 'level' in ds.dims:
        ds['level'] = ((vcoord_vals+np.roll(vcoord_vals, 1))/2)[1:]
    return ds


def _interp_vgrid(ds):
    vertical_coords = [c for c in _cosmo_vcoords if c in ds.coords]
    remap_vertical = {c: ds['vgrid'].values for c in vertical_coords}
    ds = ds.reindex(**remap_vertical, method='nearest')
    return ds


def _replace_coords(ds):
    vertical_coords = [c for c in _cosmo_vcoords if c in ds.coords]
    rename_vertical = {c: 'vgrid' for c in vertical_coords}
    ds = ds.drop(vertical_coords)
    ds = ds.rename(rename_vertical)

    rename_horizontal = {'srlat': 'rlat', 'srlon': 'rlon'}
    rename_horizontal = {k: v for k, v in rename_horizontal.items()
                         if k in ds.coords}
    ds = ds.drop(rename_horizontal.keys())
    ds = ds.rename(rename_horizontal)
    return ds


def _get_vcoord_ind(array):
    try:
        return [array[c] for c in _cosmo_vcoords if c in array.dims][0]
    except IndexError:
        return None
