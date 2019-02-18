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


def postprocess_cosmo(analysis_data, cosmo_ds):
    unstacked_analysis = analysis_data.unstack('grid')
    transposed_analysis = unstacked_analysis.transpose(
        'var_name', 'ensemble', 'time', 'vgrid', 'rlat', 'rlon'
    )
    analysis_ds = transposed_analysis.to_dataset(dim='var_name')
    pp_analysis_ds = cosmo_ds.copy(deep=True)
    for var in analysis_ds.data_vars:
        try:
            reindex_vcoord = get_vcoord_ind(pp_analysis_ds[var])
            if reindex_vcoord is None:
                reindexed_ana_var = analysis_ds[var].isel(vgrid=0)
            else:
                reindexed_ana_var = analysis_ds[var].reindex(
                    vgrid=reindex_vcoord.values, method='nearest'
                )
            pp_analysis_ds[var] = pp_analysis_ds[var].copy(
                data=reindexed_ana_var.values
            )
        except KeyError:
            logger.warning('Var: {0:s} is not found'.format(var))
    return pp_analysis_ds


def preprocess_cosmo(cosmo_ds, assim_vars):
    avail_vars = [var for var in assim_vars if var in cosmo_ds.data_vars]
    not_avail_vars = list(set(assim_vars) - set(avail_vars))
    if not_avail_vars:
        logger.warning('Following variables are not found! {0:s}'.format(
            ','.join(not_avail_vars)
        ))
    assim_ds = cosmo_ds[avail_vars]
    vgrid_ds = prepare_vgrid(assim_ds, cosmo_ds['vcoord'])
    interp_ds = interp_vgrid(vgrid_ds)
    unified_ds = replace_coords(interp_ds)
    unified_data = unified_ds.to_array(dim='var_name')
    stacked_data = unified_data.stack(grid=('rlat', 'rlon', 'vgrid'))
    if 'ensemble' not in stacked_data.dims:
        stacked_data = stacked_data.expand_dims(dim='ensemble')
    prepared_data = stacked_data.transpose('var_name', 'time', 'ensemble',
                                           'grid')
    return prepared_data


def prepare_vgrid(ds, vcoord):
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


def interp_vgrid(ds):
    vertical_coords = ['height_2m', 'height_10m', 'height_toa', 'soil1',
                       'level1', 'level']
    vertical_coords = [c for c in vertical_coords if c in ds.coords]
    remap_vertical = {c: ds['vgrid'].values for c in vertical_coords}
    ds = ds.reindex(**remap_vertical, method='nearest')
    return ds


def replace_coords(ds):
    vertical_coords = ['height_2m', 'height_10m', 'height_toa', 'soil1',
                       'level1', 'level']
    vertical_coords = [c for c in vertical_coords if c in ds.coords]
    rename_vertical = {c: 'vgrid' for c in vertical_coords}
    ds = ds.drop(vertical_coords)
    ds = ds.rename(rename_vertical)

    rename_horizontal = {'srlat': 'rlat', 'srlon': 'rlon'}
    rename_horizontal = {k: v for k, v in rename_horizontal.items()
                         if k in ds.coords}
    ds = ds.drop(rename_horizontal.keys())
    ds = ds.rename(rename_horizontal)
    return ds


def get_vcoord_ind(array):
    vertical_coords = ['height_2m', 'height_10m', 'height_toa', 'soil1',
                       'level1', 'level']
    try:
        return [array[c] for c in vertical_coords if c in array.dims][0]
    except IndexError:
        return None
