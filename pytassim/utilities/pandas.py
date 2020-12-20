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

# External modules
import pandas as pd

# Internal modules


logger = logging.getLogger(__name__)


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
