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
from typing import Iterable

# External modules
import numpy as np
import xarray as xr

# Internal modules
from . import common


logger = logging.getLogger(__name__)


_clm_vcoords = ['levsoi', 'levtot', 'levsno', 'levlak', 'no_vgrid']


def preprocess_clm(
        ds_clm: xr.Dataset,
        assim_vars: Iterable[str]
) -> xr.DataArray:
    """
    Preprocess a given CLM dataset. This dataset is typically created based
    on read-in of `clmoas.clm2.*.nc`files. Only variables specified in
    `assim_vars` are kept.
    """
    sliced_ds = ds_clm[assim_vars]
    ds_gridded = common.create_vgrid(sliced_ds, _clm_vcoords)
    ds_added_no_vgrid = common.add_no_vgrid(
        ds_gridded, _clm_vcoords, ds_gridded['vgrid'].min().values
    )
    ds_interp = _interp_vgrid(ds_added_no_vgrid)
    vertical_remap_dict = {
        'levsoi': 'vgrid', 'levlak': 'vgrid', 'no_vgrid': 'vgrid',
        'levtot': 'vgrid', 'levsno': 'vgrid',
    }

    prepared_ds = common.replace_grid(ds_interp, vertical_remap_dict)
    if 'column' in prepared_ds.dims and 'lat' not in prepared_ds.dims:
        grid_dims = ['column', 'vgrid']
    else:
        grid_dims = ['lat', 'lon', 'vgrid']
    prepared_data = common.ds_to_array(prepared_ds, grid_dims) 
    return prepared_data


def postprocess_clm(
        analysis_data: xr.DataArray,
        ds_clm: xr.Dataset
) -> xr.Dataset:
    """
    This function can be used to post-process CLM data and incorporate
    included variables into given CLM dataset. There are different steps
    included:

    1. Unstack the grid form analysis data and convert the data into a dataset
    2. Iterate through included variables and reindex these variables as they
        are in given CLM dataset
    3. Replace the variables in given CLM dataset with the analysis variables

    Parameters
    ----------
    analysis_data : :py:class:`xarray.DataArray`
        This array represents a valid analysis state, which can be a result of
        any given assimilation algorithm included in this package. In this
        analysis data, the horizontal and vertical grid are stacked together and
        the variables are a dimension within the array.
    ds_clm : :py:class:`xarray.Dataset`
        This CLM dataset is used as source dataset to convert given analysis
        array into a valid CLM dataset. The resulting dataset is a copy of
        this dataset, where the assimilated variables are replaced.

    Returns
    -------
    analysis_ds : :py:class:`xarray.Dataset`
        This analysis dataset is a copy of given CLM dataset with replaced
        variables from given analysis array.
    """
    analysis_ds = common.generic_postprocess(
        analysis_data, ds_clm, vcoords=_clm_vcoords
    )
    logger.info('Finished post-processing of CLM')
    return analysis_ds


def _interp_vgrid(
        ds: xr.Dataset
) -> xr.Dataset:
    """
    Reindexes the `vgrid` coordinate and is used in
    :py:func:`preprocess_clm`.
    """
    avail_vcoords = [c for c in _clm_vcoords if c in ds.dims]
    interp_ds = ds.reindex(
        **{c: ds['vgrid'].values for c in avail_vcoords}, method=None
    )
    return interp_ds
