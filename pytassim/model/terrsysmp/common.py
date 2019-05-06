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

# External modules
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


def create_vgrid(ds, vcoords):
    """
    Create a vertical grid based on given vertical coordinates
    """
    ds = ds.copy()
    avail_vcoords = [c for c in vcoords if c in ds.dims]
    vgrid = np.concatenate([ds[c].values for c in avail_vcoords])
    ds['vgrid'] = vgrid
    return ds


def add_no_vgrid(ds, vcoords, val=0):
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


def replace_grid(ds, remap_dict):
    """
    Replace grid variables within given dataset
    """
    rename_dict = {k: v for k, v in remap_dict.items()
                   if k in ds.coords}
    ds = ds.drop(list(rename_dict.keys()))
    ds = ds.rename(rename_dict)
    return ds


def ds_to_array(ds, grid_dims):
    """
    Stack a prepared dataset into a valid state array
    """
    unified_data = ds.to_array(dim='var_name')
    stacked_data = unified_data.stack(grid=grid_dims)
    if 'ensemble' not in stacked_data.dims:
        stacked_data = stacked_data.expand_dims(dim='ensemble')
    prepared_data = stacked_data.transpose('var_name', 'time', 'ensemble',
                                           'grid')
    return prepared_data


def array_to_ds(data, grid_dims):
    """
    Unstack an array to dataset
    """
    unstacked_data = data.unstack('grid')
    transpose_dims = ['var_name', 'ensemble', 'time'] + list(grid_dims)
    transposed_data = unstacked_data.transpose(*transpose_dims)
    prepared_ds = transposed_data.to_dataset(dim='var_name')
    return prepared_ds


def get_vert_dim(array, vcoords):
    try:
        vert_coord = [d for d in array.dims if d in vcoords][0]
    except IndexError:
        vert_coord = 'None'
    return vert_coord


def generic_postprocess(analysis_data, origin_ds, grid_dims, vcoords):
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

    Returns
    -------
    analysis_ds : :py:class:`xarray.Dataset`
        This analysis dataset is a copy of given origin dataset with replaced
        variables from given analysis array.
    """
    pre_analysis_ds = array_to_ds(
        analysis_data, grid_dims=grid_dims
    )
    analysis_ds = origin_ds.copy(deep=True)
    for var in pre_analysis_ds.data_vars:
        try:
            data_prepared = pre_analysis_ds[var].dropna('vgrid', how='all')
            dim_vert = get_vert_dim(analysis_ds[var], vcoords)
            data_prepared = data_prepared.rename({'vgrid': dim_vert})
            data_prepared = data_prepared.squeeze()
            dims_miss = set(analysis_ds[var].dims)-set(data_prepared.dims)
            for dim in dims_miss:
                data_prepared = data_prepared.expand_dims(dim)
            data_prepared = data_prepared.transpose(*analysis_ds[var].dims)
            analysis_ds[var] = analysis_ds[var].copy(
                data=data_prepared.values.reshape(analysis_ds[var].shape)
            )
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