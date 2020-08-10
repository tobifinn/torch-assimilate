#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 2/14/19
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

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


def _dot_product(x, y):
    k_mat = torch.mm(x, y.t())
    return k_mat


def gen_weights(normed_perts, normed_obs, reg_value=0):
    """
    This function is the main function to calculates the ensemble weights for
    the ensemble transform Kalman filter in the formulation of
    :cite:`hunt_efficient_2007`.
    To generate the weights, the given arguments have to be prepared and in a
    special format.
    The weights are estimated based on PyTorch.

    Parameters
    ----------
    normed_perts : :py:class:`torch.Tensor`
        The normed ensemble perturbations in observational space.
        These perturbations are estimated with observation operators and are
        normlised by their mean and the observational covariance. The shape of
        this tensor is :math:`k~x~l`, with :math:`k` as ensemble size and
        :math:`l` as observational size.
    normed_obs : :py:class:`torch.Tensor`
        The normalised observations.
        These observations are normalised by the ensemble mean in observational
        spacce and the observational covariance.
        The shape of this tensor is :math:`1~x~l`, with :math:`l` as
        observational size.

    Returns
    -------
    weights : :py:class:`torch.Tensor`
        The estimated weights, where the mean weights are added column-wise to
        the perturbations. The shape of this tensor is :math:`k~x~k`,
        with :math:`k` as ensemble size.
    w_mean : :py:class:`torch.Tensor`
        The estimated ensemble mean weights. These weights can be used to
        update the ensemble mean. The shape of this tensor is :math:`k`, the
        ensemble size.
    w_perts : :py:class:`torch.Tensor`
        The estimated ensemble perturbations in weight space. These weights
        can be used to estimate new centered ensemble perturbations. The
        shape of this tensor is :math:`k~x~k`, with :math:`k` as ensemble
        size.
    cov_analysed : :py:class:`torch.Tensor`
        The analysed covariance in weight space. This covariance can be used
        to create a posterior distribution with the mean weights. The shape of
        this tensor is :math:`k~x~k`, with :math:`k` as ensemble
        size.
    """
    kernel_perts = _dot_product(normed_perts, normed_perts)
    evals, evects, evals_inv, evects_inv = _evd(kernel_perts, reg_value)
    cov_analysed = _rev_evd(evals_inv, evects, evects_inv)
    kernel_obs = _dot_product(normed_perts, normed_obs)
    w_mean = cov_analysed @ kernel_obs
    w_perts = _det_square_root_eigen(evals_inv, evects, evects_inv)
    weights = w_mean + w_perts
    return weights, w_mean, w_perts, cov_analysed


def _evd(tensor, reg_value=0):
    evals, evects = torch.symeig(tensor, eigenvectors=True, upper=False)
    evals = evals.clamp(min=0)
    evals = evals + reg_value
    evals_inv = 1 / evals
    evects_inv = evects.t()
    return evals, evects, evals_inv, evects_inv


def _rev_evd(evals, evects, evects_inv):
    diag_flat_evals = torch.diagflat(evals)
    rev_evd = torch.mm(evects, diag_flat_evals)
    rev_evd = torch.mm(rev_evd, evects_inv)
    return rev_evd


def _det_square_root_eigen(evals_inv, evects, evects_inv):
    ens_size = evals_inv.size()[0]
    square_root_einv = ((ens_size - 1) * evals_inv).sqrt()
    w_perts = _rev_evd(square_root_einv, evects, evects_inv)
    return w_perts


def estimate_cinv(mat_to_invert):
    chol_decomp = torch.cholesky(mat_to_invert)
    chol_inv = chol_decomp.inverse()
    return chol_inv
