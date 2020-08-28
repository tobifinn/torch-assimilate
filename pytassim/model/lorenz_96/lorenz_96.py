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
from typing import Union

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


def torch_roll(a, shift, axis=0):
    """
    Shift :py:class:`torch.Tensor` along specified axis by `shift`-steps.

    Parameters
    ----------
    a : :py:class:`torch.Tensor`
        This tensor is shifted.
    shift : int
        The tensor is shifted by this number of steps. Positive values shift it
        to the right side, a negative value will shift the tensor to the left
        side.
    axis : int
        The array is shifted along this specified axis. Default is 0, which
        specifies the first axis.

    Returns
    -------
    rolled_tensor : :py:class:`torch.Tensor`
        This tensor was rolled on specified axis by ``shift``-steps. This tensor
        has same type and shape as the input tensor.
    """
    ndims = a.dim()
    left_slice = [slice(None, None)] * ndims
    left_slice[axis] = slice(-shift, None)
    right_slice = [slice(None, None)] * ndims
    right_slice[axis] = slice(None, -shift)
    rolled_tensor = torch.cat((a[left_slice], a[right_slice]), dim=axis)
    return rolled_tensor


class Lorenz96(object):
    """
    The Lorenz '96 :cite:`lorenz_predictability_1996,lorenz_optimal_1998` is a
    grid based dynamical model. In its default
    settings it has a chaotic behaviour. The grid points are wrapped in
    one-dimension such that the first and last grid point are coupled. Only
    surrounding grid points and the forcing F influence the i-th grid point at
    one time step,

    .. math::

       \\frac{dx_{i}}{dt} = (x_{i+1}-x_{i-2})\\,x_{i-1}-x_{i}+F.

    The time unit of this model equals 5 days, a typical synoptical time scale.
    The model includes internal dissipation and external forcing by the linear
    terms, while quadratic terms represent advection.

    Arguments
    ---------
    forcing : float, optional
        The forcing term in the equation. The default forcing of 8 leads to
        typical chaotic behaviour of the atmosphere.
    """
    def __init__(
            self,
            forcing: Union[float, torch.Tensor, torch.nn.Parameter] = 8.
    ):
        self.forcing = forcing

    def __str__(self):
        return 'Lorenz96(F={0})'.format(self.forcing)

    def __repr__(self):
        return 'Lorenz96'

    @staticmethod
    def _calc_advection(state: torch.Tensor) -> torch.Tensor:
        """
        This method calculates the advection term of the Lorenz '96 model. This
        term is given by

        .. math::

           (x_{i+1}-x_{i-2})\\,x_{i-1}.

        Parameters
        ----------
        state : :py:class:`torch.Tensor`
            This state is used to calculate the advection term of the Lorenz
            model. The last axis should be the grid axis along which the
            advection is calculated.

        Returns
        -------
        advection : :py:class:`torch.Tensor`
            The calculated advection based on given state. The advection has the
            same type and shape as the input state.
        """
        diff = torch_roll(state, -1, axis=-1) - torch_roll(state, 2, axis=-1)
        advection = diff * torch_roll(state, 1, axis=-1)
        return advection

    def _calc_dissipation(self, state: torch.Tensor) -> torch.Tensor:
        """
        This method calculates the dissipation term. The term is given by

        .. math::

            -x_{i}.

        Parameters
        ----------
        state : :py:class:`torch.Tensor`
            This state is used to calculate the dissipation term of the Lorenz
            model.

        Returns
        -------
        dissipation : :py:class:`torch.Tensor`
            The calculated dissipation based on given state. The dissipation has
            same type and shape as the input state.
        """
        dissipation = -state
        return dissipation

    def _calc_forcing(self, state: torch.Tensor) -> torch.Tensor:
        """
        This method calculates the forcing. This returns set forcing, which
        a constant forcing in Lorenz '96 model. This method can be overwritten
        to introduce a coupling between different models. The forcing is
        currently given by

        .. math::

            F.

        Parameters
        ----------
        state : :py:class:`torch.Tensor`
            This state is currently not used, but can be used to introduce a
            state-dependent forcing.

        Returns
        -------
        forcing : :py:class:`torch.Tensor` or float
            The calculated forcing based on attributes and given state. If it is
            only a float than the same forcing is used at every grid point.
        """
        forcing = self.forcing
        return forcing

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculates the time-derivative :math:`\\frac{dx_{i}}{dt}` for given
        state.

        Parameters
        ----------
        state : :py:class:`torch.Tensor`
            This state is used to estimated the current time-derivative. The
            last axis is supposed to be the grid axis.

        Returns
        -------
        derivative : :py:class:`torch.Tensor`
            The calculated time-derivative based on given state. This
            time-derivative has the same type and shape as given state.
        """
        advection = self._calc_advection(state)
        dissipation = self._calc_dissipation(state)
        forcing = self._calc_forcing(state)

        derivative = advection + dissipation + forcing
        return derivative
