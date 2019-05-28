#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 5/28/19
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
import io

# External modules
from distributed.protocol import dask_serialize, dask_deserialize, serialize, \
    register_serialization_family, register_serialization
import dask.array as da

import torch.jit
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


def bag_to_array(bag_to_transform, shape):
    arr_list = [da.from_delayed(ele, shape=shape, dtype=np.float64)
                for ele in bag_to_transform]
    out_array = da.stack(arr_list, axis=0).rechunk(-1)
    return out_array
