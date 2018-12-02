#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 02.12.18
#
# Created for torch-assim
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2018}  {Tobias Sebastian Finn}
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


def torch_roll(x, shift=1):
    """
    Roll :py:class:`torch.Tensor` along first axis by `shift`-steps.

    Parameters
    ----------
    x : :py:class:`torch.Tensor`
        This tensor is rolled.
    shift : int, optional
        The tensor is rolled by this number of steps. Normally it is rolled to
        the right side, a negative value will roll the tensor to the left side.

    Returns
    -------
    rolled_tensor : :py:class:`torch.Tensor`
        This tensor was rolled on first axis by ``shift``-steps.
    """
    rolled_tensor = torch.cat((x[-shift:], x[:-shift]))
    return rolled_tensor


class Lorenz96(object):
    """
    The Lorenz '96 [L96]_ is a grid based dynamical model. In its default
    settings it has a chaotic behaviour. The grid points are wrapped in
    one-dimension such that the first and last grid point are coupled. Only
    surrounding grid points and the forcing F influence the i-th grid point at
    one time step,

    .. math::

       \\frac{dx_{i}}{dt} = (x_{i+1}-x_{i-2})\\,x_{i-1}-x_{i}+F.

    The time unit of this model equals 5 days, a typical synoptical time scale.
    It includes internal dissipation and external forcing by the linear terms,
    while quadratic terms lead to advection.

    .. [L96] Lorenz, E. N. (1996, September). Predictability: A problem partly
           solved. In Proc. Seminar on predictability (Vol. 1, No. 1).
    .. [L98] Lorenz, E. N., & Emanuel, K. A. (1998). Optimal sites for
           supplementary weather observations: Simulation with a small model.
           Journal of the Atmospheric Sciences, 55(3), 399-414.

    Arguments
    ---------
    forcing : float, optional
        The forcing term in the equation. The default forcing of 8 leads to
        typical chaotic behaviour of the atmosphere.
    nr_points : int, optional
        The number of grid points of this model. The whole model spans an
        atmospheric corridor around the world such that the first and last grid
        point are connected. The default value is 40 grid points as in [L98]_.
    """
    def __init__(self, forcing=8, nr_points=40,):
        self.forcing = forcing
        self.nr_points = nr_points

    def __call__(self, state):
        pass
