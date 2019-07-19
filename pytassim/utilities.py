#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 7/17/19
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


def chol_solve(x, y, alpha=0):
    """
    Estimate the least squares solution of
    :math:`(X^T X + \\alpha I)^{-1} X^T Y` with cholesky decomposition and
    tikhonov regularization in form of a L2-penalty. If the covariance of
    :math:`x` cannot be decomposed, the regularization factor alpha will be
    automatically increased.

    Parameters
    ----------
    x : :py:class:`torch.Tensor`
        This is the input tensor, which are also called predictors.
    y : :py:class:`torch.Tensor`
        This is the target tensor such that :math:`x \\beta` is as near as
        possible to this tensor in the least-squares sense.
    alpha : float, optional
        This is the regularization factor. An increase of :math:`\\alpha` is
        also an increase of the regularization. :math:`\\alpha = 0` deactivates
        the regularization (default). If the data covariance cannot be
        decomposed this regularization factor will be increased.

    Returns
    -------
    calc_beta : :py:class:`torch.Tensor`
        The calculated :math:`\\beta`.

    Raises
    ------
    ValueError
        A ValueError is raised if no convergence of the cholesky iterations is
        possible. The stop criterion is an alpha above 10E30.
    """
    x_t = x.t()
    cov_xx = torch.matmul(x_t, x)
    cov_xy = torch.matmul(x_t, y)
    mat_size = cov_xx.size()[1]
    step = mat_size + 1
    end = mat_size * mat_size
    calc_beta = None
    while calc_beta is None:
        try:
            print(cov_xx)
            mat_upper = torch.cholesky(cov_xx, upper=True)
            calc_beta = torch.potrs(cov_xy, mat_upper, upper=True).t()
        except RuntimeError:
            cov_xx.view(-1)[:end:step] -= alpha
            if alpha == 0:
                alpha = 0.00001
            else:
                alpha *= 10
            cov_xx.view(-1)[:end:step] += alpha
        if alpha > 10E30 and calc_beta is None:
            raise ValueError(
                'No convergence for cholesky decomposition possible!'
            )
    logger.debug('Cholesky decomposition alpha: {0:.2E}'.format(alpha))
    return calc_beta
