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
import torch
import dask.array as da
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


def evd(
        tensor: torch.Tensor,
        reg_value: torch.Tensor = torch.tensor(0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs eigendecomposition of a symmetric hermitian tensor. The eigenvalues
    of the nearest positive semidefinit matrix are returned to ensure
    differentiability.

    Parameters
    ----------
    tensor : :py:class:`torch.Tensor` (..., nx, nx)
        This tensor is eigen decomposed. The values of the nearest positive
        semidefinit matrix to this tensor are returned.
    reg_value : :py:class:`torch.Tensor`, optional
        This regularization value is added to the eigenvalues and represents
        a regularization of the matrix diagonal. The regularization is
        deactivated with the default value of 0.

    Returns
    -------
    evals : :py:class:`torch.Tensor` (..., nx)
        The eigenvalues of the nearest positive semidefinit matrix to the
        given tensor. The regularization value is already added to these
        eigenvalues.
    evects : :py:class:`torch.Tensor` (..., nx, nx)
        The estimated eigenvectors based on given tensor.
    evals_inv : :py:class:`torch.Tensor` (..., nx)
        The inverted eigenvalues, which can be used to invert the given tensor.
    """
    evals, evects = torch.symeig(tensor, eigenvectors=True, upper=False)
    evals = evals.clamp(min=0)
    evals = evals + reg_value
    evals_inv = 1 / evals
    return evals, evects, evals_inv


def rev_evd(
        evals: torch.Tensor,
        evects: torch.Tensor,
) -> torch.Tensor:
    """
    Composes a tensor :math:`A` based on given eigenvalues :math:`\\lambda` and
    eigenvectors :math:`u`,


    .. math::

       A = u \\lambda (u)^T


    Parameters
    ----------
    evals : :py:class:`torch.Tensor` (..., nx)
        These eigenvalues are used to recompose the matrix.
    evects : :py:class:`torch.Tensor` (..., nx, nx)
        These eigenvectors are used to recompose the matrix.

    Returns
    -------
    rev_mat : :py:class:`torch.Tensor` (..., nx, nx)
        The recomposed matrix based on given eigenvalues and eigenvectors.
    """
    diag_flat_evals = torch.diag_embed(evals)
    rev_mat = torch.einsum('...ij,...jk->...ik', evects, diag_flat_evals)
    rev_mat = torch.einsum('...ij,...kj->...ik', rev_mat, evects)
    return rev_mat


def grid_to_array(
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
