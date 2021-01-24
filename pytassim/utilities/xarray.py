#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 24.01.21
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union, Any

# External modules
import xarray as xr
import pandas as pd
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


__all__ = [
    'save_netcdf',
    'load_netcdf',
    'encode_multidim',
    'decode_multidim'
]


def save_netcdf(
        dataset_to_save: Union[xr.DataArray, xr.Dataset],
        save_path: Any = None,
        *args,
        **kwargs
) -> Union[bytes, "Delayed", None]:
    """
    Save a given dataset or dataarray to a netCDF-file. MultiDimensional
    coordinates are converted into raw coordinate value.

    Parameters
    ----------
    dataset_to_save : xarray.DataArray or xarray.Dataset
        This DataArray or Dataset is stored under given save path.
    save_path : str, Path or file-like, optional
        The file is stored under this save path. For further information,
        please take a look into the documentation of
        :py:meth:``xarray.Dataset.to_netcdf``.
    *args
        These additional arguments are passed to
        :py:meth:``xarray.Dataset.to_netcdf``.
    **kwargs
        These additional keyword arguments are passed to
        :py:meth:``xarray.Dataset.to_netcdf``.
    """
    dataset_to_save = encode_multidim(
        dataset=dataset_to_save
    )
    return dataset_to_save.to_netcdf(save_path, *args, **kwargs)


def encode_multidim(
        dataset: Union[xr.DataArray, xr.Dataset],
) -> Union[xr.DataArray, xr.Dataset]:
    """
    This function encodes multidimensional indexes from given dataarray or
    dataset into single dimensional indexes.
    The dropped index levels are converted to raw coordinate values.
    The single dimensional index has than an additional attribute
    `multidim_levels`, which indicates the order and names of the
    multidimensional index levels.

    Parameters
    ----------
    dataset : xarray.DataArray or xarray.Dataset
        The multidimensional indexes of given dataarray or dataset are
        encoded into single dimensional indexes.

    Returns
    -------
    dataset : xarray.DataArray or xarray.Dataset
        A copy of the given object with encoded multidimensional indexes. If
        no multidimensional index is found, the original object is returned.
    """
    multidim_dims = [
        dim for dim in dataset.dims
        if isinstance(dataset.indexes[dim], pd.MultiIndex)
    ]
    for dim in multidim_dims:
        levels_str = ';'.join(dataset.indexes[dim].names)
        dataset = dataset.reset_index(dim)
        dataset = dataset.assign_coords(
            {dim: np.arange(len(dataset[dim]))}
        )
        dataset[dim].attrs['multidim_levels'] = levels_str
    return dataset


def load_netcdf(
        load_path: Any,
        array: bool = False,
        *args, **kwargs
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Loads a dataset or dataarray from a given load path.
    Single-dimensional indexes with 'multidim_levels' as attribute
    will be converted into MultiDimensional indexes.

    Parameters
    ----------
    load_path : str, Path, file-like or DataStore
        The dataset or dataarray will be loaded from this path. For further
        information please look into the documentation of
        :py:meth:``xarray.open_dataset``.
    array : bool, optional
        If either a dataarray should be loaded or a dataset (default).
    *args
        These additional arguments are passed to
        :py:meth:``xarray.open_dataset``.
    **kwargs
        These additional keyword arguments are passed to
        :py:meth:``xarray.open_dataset``.
    """
    if array:
        loaded_dataset = xr.open_dataarray(load_path, *args, **kwargs)
    else:
        loaded_dataset = xr.open_dataset(load_path, *args, **kwargs)
    loaded_dataset = decode_multidim(dataset=loaded_dataset)
    return loaded_dataset


def decode_multidim(
        dataset: Union[xr.DataArray, xr.Dataset],
) -> Union[xr.DataArray, xr.Dataset]:
    """
    This function decodes single-dimensional indexes into multi-dimensional
    indexes, if 'multidim_levels' is found within the attributes of the index.

    Parameters
    ----------
    dataset : xarray.DataArray or xarray.Dataset
        The dimensions of this given dataarray or dataset are converted into
        multidimensional indexes.

    Returns
    -------
    dataset : xarray.DataArray or xarray.Dataset
        A copy of the given dataarray or dataset with indexes converted into
        multidimensional indexes, where possible.
        If no multidimensional index indicator is found, the given object is
        returned.
    """
    multidim_dims = [
        dim for dim in dataset.dims
        if 'multidim_levels' in dataset[dim].attrs.keys()
    ]
    for dim in multidim_dims:
        level_names = dataset[dim].attrs['multidim_levels'].split(';')
        dim_multiindex = pd.MultiIndex.from_arrays(
            [dataset[level_name].values for level_name in level_names],
            names=level_names
        )
        attrs = dataset[dim].attrs
        _ = attrs.pop('multidim_levels')
        dataset = dataset.reset_coords(level_names, drop=True)
        dataset = dataset.assign_coords({dim: dim_multiindex})
        dataset[dim].attrs = attrs
    return dataset
