#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 11.12.20
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Tuple

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


def evd(
        tensor: torch.Tensor,
        reg_value: float = 0.,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs eigendecomposition of a symmetric hermitian tensor. The eigenvalues
    of the nearest positive semidefinit matrix are returned to ensure
    differentiability.

    Parameters
    ----------
    tensor : :py:class:`torch.Tensor` (nx, nx)
        This tensor is eigen decomposed. The values of the nearest positive
        semidefinit matrix to this tensor are returned.
    reg_value : float, optional
        This regularization value is added to the eigenvalues and represents
        a regularization of the matrix diagonal. The regularization is
        deactivated with the default value of 0.

    Returns
    -------
    evals : :py:class:`torch.Tensor` (nx)
        The eigenvalues of the nearest positive semidefinit matrix to the
        given tensor. The regularization value is already added to these
        eigenvalues.
    evects : :py:class:`torch.Tensor` (nx, nx)
        The estimated eigenvectors based on given tensor.
    evals_inv : :py:class:`torch.Tensor` (nx)
        The inverted eigenvalues of the nearest positive semidefinit matrix
        to the given tensor.
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
    evals : :py:class:`torch.Tensor` (nx)
        These eigenvalues are used to recompose the matrix.
    evects : :py:class:`torch.Tensor` (nx, nx)
        These eigenvectors are used to recompose the matrix.

    Returns
    -------
    rev_mat : :py:class:`torch.Tensor` (nx, nx)
        The recomposed matrix based on given eigenvalues and eigenvectors.
    """
    diag_flat_evals = torch.diag_embed(evals)
    rev_mat = torch.mm(evects, diag_flat_evals)
    rev_mat = torch.mm(rev_mat, evects.t())
    return rev_mat


def dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Helper function for dot product :math:`x \\cdot y`. This dot product is
    outsourced due to performance considerations if multi-dimensional values
    are given.

    Parameters
    ----------
    x : torch.Tensor
        The first input to the dot product with (..., k, l) as shape.
    y : torch.Tensor
        The second input to the dot product with (..., m, l) as shape. The last
        two dimensions are transposed to estimate the dot product.

    Returns
    -------
    product : torch.Tensor
        The dot product with (..., k, m) as shape.
    """
    product = torch.bmm(
        x.reshape(-1, *x.shape[-2:]),
        y.transpose(-1, -2).reshape(-1, y.shape[-1], y.shape[-2])
    )
    product = product.view(*x.shape[:-2], x.shape[-2], y.shape[-2])
    return product
