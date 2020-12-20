#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 20.12.20
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Any

# External modules
import pandas as pd
import numpy as np
import dask.array as da

# Internal modules


logger = logging.getLogger(__name__)


def dtindex_to_total_seconds(index: pd.DatetimeIndex) -> pd.Index:
    """
    Function to convert a `pandas.DatetimeIndex` into a `pd.Index` with the
    total seconds since 1970-01-01 (unix time) as float values.

    Parameters
    ----------
    index : pd.DatetimeIndex
        This index is converted into a float index.

    Returns
    -------
    total_seconds : pd.Index
        The converted index with the unix time as value.
    """
    index_diff = index - pd.Timestamp(1970, 1, 1)
    total_seconds = index_diff.total_seconds()
    return total_seconds


def multiindex_to_frame(multiindex: pd.MultiIndex) -> pd.DataFrame:
    """
    Function to convert a `pandas.MultiIndex` into a `pandas.DataFrame`.
    The automatically created index of the dataframe is dropped in favor of
    an integer-based index.

    Parameters
    ----------
    multiindex : pd.MultiIndex
        This multiindex is converted into a dataframe. The names of the
        multiindex are used as columns.

    Returns
    -------
    index_frame : pd.DataFrame
        The constructed dataframe based on given multiindex.
    """
    index_frame = multiindex.to_frame()
    index_frame = index_frame.reset_index(drop=True)
    return index_frame


def index_to_array(
        index: Any
) -> np.ndarray:
    """
    Transform a given index into a :py:class:`np.ndarray`, which can be then
    used by localization within the assimilation.

    Parameters
    ----------
    index : any
        This index is transformed into an array.

    Returns
    -------
    index_array : :py:class:`np.ndarray` (n_points, n_grid)
        This is the transformmed array, which is then used within the
        assimilation.
    """
    if isinstance(index, np.ndarray):
        raw_index_array = index
    elif isinstance(index, da.Array):
        raw_index_array = index.compute()
    else:
        raw_index_array = np.atleast_1d(index.values)
    if isinstance(raw_index_array[0], tuple):
        shape = (-1, len(raw_index_array[0]))
    elif raw_index_array.ndim > 1:
        shape = raw_index_array.shape
    else:
        shape = (-1, 1)
    dtype = ','.join(['float']*shape[-1])
    index_array = np.array(index, dtype=dtype).view(float).reshape(*shape)
    return index_array
