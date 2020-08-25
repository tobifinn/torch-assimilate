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

# External modules
import torch
import dask.array as da
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


def evd(tensor, reg_value=torch.tensor(0)):
    evals, evects = torch.symeig(tensor, eigenvectors=True, upper=False)
    evals = evals.clamp(min=0)
    evals = evals + reg_value
    evals_inv = 1 / evals
    return evals, evects, evals_inv


def rev_evd(evals, evects):
    diag_flat_evals = torch.diag_embed(evals)
    rev_mat = torch.einsum('...ij,...jk->...ik', evects, diag_flat_evals)
    rev_mat = torch.einsum('...ij,...kj->...ik', rev_mat, evects)
    return rev_mat


def grid_to_array(index):
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
