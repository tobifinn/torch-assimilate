#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 03.08.20
#
# Created for 20_kernel_etkf
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

# Internal modules


logger = logging.getLogger(__name__)


def dot_product(
        x: torch.Tensor,
        y: torch.Tensor
) -> torch.Tensor:
    """
    The linear kernel defined by the dot product between x and y.

    Parameters
    ----------
    x : :py:class:`torch.Tensor` (n_samples_x, n_features)
        The first input to the kernel.
    y : :py:class:`torch.Tensor` (n_samples_y, n_features)
        The second input to the kernel.

    Returns
    -------
    mat : :py:class:`torch.Tensor` (n_samples_x, n_samples_y)
        The dot product between given `x` and `y`.
    """
    mat = torch.einsum('...ij,...kj->...ik', x, y)
    return mat


def distance_matrix(
        x: torch.Tensor,
        y: torch.Tensor,
        norm: float = 2.
) -> torch.Tensor:
    """
    The distance matrix between given tensors defined via the p-norm distance.

    Parameters
    ----------
    x : :py:class:`torch.Tensor` (..., n_samples_x, n_features)
        The first input to the kernel.
    y : :py:class:`torch.Tensor` (..., n_samples_y, n_features)
        The second input to the kernel.
    norm : :py:class:`float`
        The norm of this distance (default=2.0).

    Returns
    -------
    dist : :py:class:`torch.Tensor` (..., n_samples_x, n_samples_y)
        p-norm distance between x and y.
    """
    x_batched = x.unsqueeze(0)
    y_batched = y.unsqueeze(0)
    dist_tensor = torch.cdist(x_batched, y_batched, p=norm)
    dist_squeezed = dist_tensor.squeeze(0)
    return dist_squeezed


def euclidean_dist(
        x: torch.Tensor,
        y: torch.Tensor
) -> torch.Tensor:
    """
    The euclidean distance defined as squared difference between x and y.

    Parameters
    ----------
    x : :py:class:`torch.Tensor` (..., n_samples_x, n_features)
        The first input to the kernel.
    y : :py:class:`torch.Tensor` (..., n_samples_y, n_features)
        The second input to the kernel.

    Returns
    -------
    dist : :py:class:`torch.Tensor` (..., n_samples_x, n_samples_y)
        Euclidean distance between x and y.
    """
    dist = distance_matrix(x, y, norm=2.).pow(2)
    return dist
