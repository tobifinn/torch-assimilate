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
from typing import Union

# External modules
import xarray as xr
import pandas as pd
import numpy as np

import dask

# Internal modules


logger = logging.getLogger(__name__)


__all__ = [
    'save_netcdf',
    'load_netcdf',
    'encode_multidim_dataset',
    'decode_mutlidim_dataset'
]


def save_netcdf(
        dataset_to_save: Union[xr.DataArray, xr.Dataset],
        save_path: str
) -> Union[bytes, "Delayed", None]:
    if isinstance(dataset_to_save, xr.DataArray):
        dataset_to_save = dataset_to_save.to_dataset(
            name='__xarray_dataarray_variable__'
        )
    dataset_to_save = encode_multidim_dataset(
        dataset=dataset_to_save
    )
    return dataset_to_save.to_netcdf(path=save_path)


def encode_multidim_dataset(
        dataset: xr.Dataset
) -> xr.Dataset:
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


def load_netcdf(load_path: str):
    pass


def decode_mutlidim_dataset(
        dataset: xr.Dataset
) -> xr.Dataset:
    pass
