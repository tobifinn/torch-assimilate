#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 2/21/19
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
from typing import Iterable, Dict, Any

# External modules
import numpy as np
import xarray as xr

# Internal modules


logger = logging.getLogger(__name__)


def create_vgrid(
        ds: xr.Dataset,
        vcoords: Iterable[str]
) -> xr.Dataset:
    """
    Create a vertical grid based on given vertical coordinates
    """
    ds = ds.copy()
    avail_vcoords = [c for c in vcoords if c in ds.dims]
    vgrid = np.concatenate([ds[c].values for c in avail_vcoords])
    ds['vgrid'] = vgrid
    return ds


def add_no_vgrid(
        ds: xr.Dataset,
        vcoords: Iterable[str],
        val: float = 0
) -> xr.Dataset:
    """
    Add an additional variable if no vertical grid is available
    """
    ds = ds.copy()
    vars_wo_vgrid = [var for var in ds.data_vars
                     if set(ds[var].dims).isdisjoint(vcoords)]
    for var in vars_wo_vgrid:
        ds[var] = ds[var].expand_dims('no_vgrid', axis=-3)
    if vars_wo_vgrid:
        ds['no_vgrid'] = np.array([val, ])
    return ds


def replace_grid(
        ds: xr.Dataset,
        remap_dict: Dict[str, Any]
) -> xr.Dataset:
    """
    Replace grid variables within given dataset
    """
    rename_dict = {k: v for k, v in remap_dict.items()
                   if k in ds.coords}
    ds = ds.drop_sel(list(rename_dict.keys()))
    ds = ds.rename(rename_dict)
    return ds


def ds_to_array(ds: xr.Dataset, grid_dims: Iterable[str]) -> xr.DataArray:
    """
    Stack a prepared dataset into a valid state array
    """
    unified_data = ds.to_array(dim='var_name')
    stacked_data = unified_data.stack(grid=grid_dims)
    if 'ensemble' not in stacked_data.dims:
        stacked_data = stacked_data.expand_dims(dim='ensemble')
    transpose_dims = ['var_name', 'time', 'ensemble', 'grid']
    transpose_dims = [d for d in transpose_dims if d in stacked_data.dims]
    prepared_data = stacked_data.transpose(*transpose_dims)
    return prepared_data


def array_to_ds(data: xr.DataArray) -> xr.Dataset:
    """
    Unstack an array to dataset
    """
    unstacked_data = data.unstack('grid')
    grid_dims = list(data.indexes['grid'].names)
    transpose_dims = ['var_name', 'ensemble', 'time'] + list(grid_dims)
    transpose_dims = [d for d in transpose_dims if d in unstacked_data.dims]
    transposed_data = unstacked_data.transpose(*transpose_dims)
    prepared_ds = transposed_data.to_dataset(dim='var_name')
    return prepared_ds


def dim_transpose(array: xr.DataArray, vcoords: Iterable[str]):
    dim_generic = ['time', 'ensemble']
    dim_order = [d for d in dim_generic if d in array.dims]
    dim_order += [d for d in vcoords+['vgrid', ] if d in array.dims]
    dim_grid = [d for d in array.dims if d not in dim_order]
    dim_order += dim_grid
    array_trans = array.transpose(*dim_order)
    return array_trans


def generic_postprocess(
        analysis_data: xr.DataArray,
        origin_ds: xr.Dataset,
        vcoords: Iterable[str]
) -> xr.Dataset:
    """
    This function can be used to post-process analysis data and incorporate
    included variables into given origin dataset. There are different steps
    included:

    1. Unstack the grid form analysis data and convert the data into a dataset
    2. Iterate through included variables and reindex these variables as they
        are in given origin dataset
    3. Replace the variables in given origin dataset with the analysis variables

    Parameters
    ----------
    analysis_data : :py:class:`xarray.DataArray`
        This array represents a valid analysis state, which can be a result of
        any given assimilation algorithm included in this package. In this
        analysis data, the horizontal and vertical grid are stacked together and
        the variables are a dimension within the array.
    origin_ds : :py:class:`xarray.Dataset`
        This origin dataset is used as source dataset to convert given analysis
        array into a valid origin dataset. The resulting dataset is a copy of
        this dataset, where the assimilated variables are replaced.
    vcoords : Iterable(str)
        These vcoords are used to unstack the analysis data.

    Returns
    -------
    analysis_ds : :py:class:`xarray.Dataset`
        This analysis dataset is a copy of given origin dataset with replaced
        variables from given analysis array.
    """
    pre_analysis_ds = array_to_ds(analysis_data)
    logger.info('Created analysis dataset')
    analysis_ds = origin_ds.copy(deep=True)
    logger.info('Copied original dataset')
    for var in pre_analysis_ds.data_vars:
        logger.info('Starting to post-process {0:s}'.format(var))
        try:
            data_prepared = pre_analysis_ds[var].dropna('vgrid', how='all')
            data_prepared = dim_transpose(data_prepared, vcoords)
            tmp_analysis_var = dim_transpose(analysis_ds[var], vcoords)
            tmp_analysis_var.data = data_prepared.data.reshape(
                tmp_analysis_var.shape
            )
            analysis_ds[var] = tmp_analysis_var.transpose(
                *analysis_ds[var].dims
            )
            logger.info('Post-processed {0:s}'.format(var))
        except KeyError:
            logger.warning('Var: {0:s} is not found'.format(var))
        except ValueError:
            logger.warning(
                'Var: {0:s} is not broadcastable ({1:s} != {2:s})'.format(
                    var, str(data_prepared.shape),
                    str(analysis_ds[var].shape)
                )
            )
    return analysis_ds
