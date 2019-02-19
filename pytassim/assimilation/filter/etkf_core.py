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


def gen_weights(back_prec, innov, hx_perts, obs_cov, obs_weights=1):
    """
    This function is the main function to calculates the ensemble weights,
    based on cite:`hunt_efficient_2007`. To generate the weights, the given
    arguments have to be
    prepared and in a special format. The weights are estimated with
    PyTorch.

    Parameters
    ----------
    innov : :py:class:`torch.tensor`
        These innovations are multiplied by the ensemble gain to estimate
        the mean ensemble weights. These innovation should have a shape of
        :math:`l`, the observation length.
    hx_perts : :py:class:`torch.tensor`
        These are the ensemble perturbations in ensemble space. These
        perturbations are used to calculated the analysed ensemble
        covariance in weight space. These perturbations have a shape of
        :math:`l~x~k`, with :math:`k` as ensemble size and :math:`l` as
        observation length.
    obs_cov : :py:class:`torch.tensor`
        This tensor represents the observation covariance. This covariance
        is used for the estimation of the analysed covariance in weight
        space. The shape of this covariance should be :math:`l~x~l`, with
        :math:`l` as observation length.
    back_prec : :py:class:`torch.tensor`
        This normalized background precision is an identity matrix scaled by
        :math:`(k-1)`, with :math:`k` as ensemble size.
    obs_weights : :py:class:`torch.tensor` or float, optional
        These are the observation weights. These observation weights can be
        used for localization or weighting purpose. If these observation
        weights are a float, then the same weight for every observation is
        used. If these weights are a :py:class:`~torch.tensor`, then the
        shape of this tensor should be :math:`l`, the observation length.
        The default values is 1, indicating that every observation is
        uniformly weighted.

    Returns
    -------
    w_mean : :py:class:`torch.tensor`
        The estimated ensemble mean weights. These weights can be used to
        update the ensemble mean. The shape of this tensor is :math:`k`, the
        ensemble size.
    w_perts : :py:class:`torch.tensor`
        The estimated ensemble perturbations in weight space. These weights
        can be used to estimate new centered ensemble perturbations. The
        shape of this tensor is :math:`k~x~k`, with :math:`k` as ensemble
        size.
    """
    if len(innov.size()) == 0:
        ens_size = back_prec.shape[0]
        w_mean = torch.zeros(ens_size, dtype=innov.dtype)
        w_perts = torch.eye(ens_size, dtype=innov.dtype)
        return w_mean, w_perts
    estimated_c = _compute_c(hx_perts, obs_cov, obs_weights)
    prec_ana = _calc_precision(estimated_c, hx_perts, back_prec)
    evd = _eigendecomp(prec_ana)
    evals, evects, evals_inv, evects_inv = evd

    cov_analysed = torch.matmul(evects, torch.diagflat(evals_inv))
    cov_analysed = torch.matmul(cov_analysed, evects_inv)

    gain = torch.matmul(cov_analysed, estimated_c)
    w_mean = torch.matmul(gain, innov)

    w_perts = _det_square_root_eigen(evals_inv, evects, evects_inv)
    return w_mean, w_perts


def _eigendecomp(precision):
    evals, evects = torch.symeig(precision, eigenvectors=True, upper=False)
    evals[evals < 0] = 0
    evals_inv = 1 / evals
    evects_inv = evects.t()
    return evals, evects, evals_inv, evects_inv


def _det_square_root_eigen(evals_inv, evects, evects_inv):
    ens_size = evals_inv.size()[0]
    w_perts = ((ens_size - 1) * evals_inv) ** 0.5
    w_perts = torch.matmul(evects, torch.diagflat(w_perts))
    w_perts = torch.matmul(w_perts, evects_inv)
    return w_perts


def _calc_precision(c, hx_perts, back_prec):
    prec_obs = torch.matmul(c, hx_perts)
    prec_ana = back_prec + prec_obs
    return prec_ana


def _compute_c(hx_perts, obs_cov, obs_weights=1):
    if torch.allclose(obs_cov, torch.diag(torch.diagonal(obs_cov))):
        calculated_c = _compute_c_diag(hx_perts, obs_cov)
    else:
        calculated_c, _ = _compute_c_chol(hx_perts, obs_cov)
    calculated_c = calculated_c * obs_weights
    return calculated_c


def _compute_c_diag(hx_perts, obs_cov):
    calculated_c = hx_perts.t() / torch.diagonal(obs_cov)
    return calculated_c


def _compute_c_chol(hx_perts, obs_cov, alpha=0):
    obs_cov_prod = torch.matmul(obs_cov.t(), obs_cov)
    obs_hx = torch.matmul(obs_cov.t(), hx_perts)
    mat_size = obs_cov_prod.size()[1]
    step = mat_size + 1
    end = mat_size * mat_size
    calculated_c = None
    while calculated_c is None:
        try:
            mat_upper = torch.cholesky(obs_cov_prod, upper=True)
            calculated_c = torch.potrs(obs_hx, mat_upper, upper=True).t()
        except RuntimeError:
            obs_cov_prod.view(-1)[:end:step] -= alpha
            if alpha == 0:
                alpha = 0.00001
            else:
                alpha *= 10
            obs_cov_prod.view(-1)[:end:step] += alpha
    return calculated_c, alpha
