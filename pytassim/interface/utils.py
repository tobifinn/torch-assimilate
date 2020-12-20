#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 10.08.20
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}
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
from typing import Tuple, Any

# External modules
import pandas as pd
import dask.array as da
import numpy as np

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
