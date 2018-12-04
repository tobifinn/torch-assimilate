#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12/4/18
#
# Created for torch-assimilate
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


class Lorenz84(object):
    """
    Lorenz '84 [L84]_ is a model with three variables. These three variables are
    coupled such that the Hadley circulation is emulated. The `X` variable
    represents the amplitude of the westerly wind current, while the other two
    variables (`Y` and `Z`) show the cosine and sine phase of superimposed
    large-scale eddies. They transport heat from tropics towards the poles.
    Considering these terms, we can construct following time derivative of the
    westerly current

    .. math:: \\frac{dX}{dt} = -Y^2 - Z^2 - aX + aF.


    The amplification of the eddies slows the westerly current, as
    represented in :math:`-Y^2` and :math:`-Z^2`. Mechanical and thermal damping
    decreases the eddy phase and slows the westerly current as shown by the
    linear term :math:`-X`. The damping factor :math:`a` controls the damping
    speed of the westerly current, which is also coupled to the symmetric
    forcing :math:`F`.

    The westerly current is coupled to the phase of the eddies

    .. math::

       \\frac{dY}{dt} &= XY - bXZ - Y + G,

       \\frac{dZ}{dt} &= bXY + XZ - Z.

    The time-derivative of the cosine :math:`\\frac{dY}{dt}` and sine
    :math:`\\frac{dZ}{dt}` phase of eddies is influenced by the amplification of
    eddies caused by interactions with the westerly current :math:`X`. The
    factor :math:`b` in front shows eddy displacement due to advection by the
    current. The asymmetric forcing :math:`G` increases the cosine phase of
    eddies, which is then propagated to westerly current and sine phase via the
    coupling of the three variables.

    The typical time unit of this model is 5 days. The default values of the
    arguments are set such that the model represent typical meteorological
    instability [L84]_.

    Arguments
    ---------
    damp_factor : float, optional
        The damping factor :math:`a` of the westerly current. This factor
        controls the damping of the current and the influence of thermal
        forcing. Default value is 0.25 such that the current is slower damped
        than the eddies.
    dis_factor : float, optional
        The displacement factor :math:`b`, controlling the displacement of the
        eddies by the westerly current. Default value is 4.0, leading to more
        rapid displacement of the eddies than amplification.
    symm_forcing : float, optional
        The symmetric thermal forcing :math:`F` of the westerly current. This
        forcing increases the amplification of the westerly current. If the
        variables were not coupled the amplification would converge to this
        value. Default value is 8.
    asymm_forcing : float, optional
        The asymmetric thermal forcing :math:`G` of the eddies. This forcing
        influences the eddies. :math:`Y` would converge to this value if the
        variables are not coupled by other terms. Default value is 1.

    References
    ----------
    .. [L84] Lorenz, E. N. (1984). Irregularity: a fundamental property of the
            atmosphere. Tellus A, 36A(2), 98–110.
            https://doi.org/10.1111/j.1600-0870.1984.tb00230.x
    """
    def __init__(self, damp_factor=0.25, dis_factor=4.0, symm_forcing=8.0,
                 asymm_forcing=1.0):
        self.dis_factor = dis_factor
        self.damp_factor = damp_factor
        self.symm_forcing = symm_forcing
        self.asymm_forcing = asymm_forcing

    def _calc_westerly(self, state):
        coupling = -state[..., 1] ** 2 - state[..., 2] ** 2
        damping = self.damp_factor * state[..., 0]
        forcing = self.damp_factor * self.symm_forcing
        amp = coupling - damping + forcing
        return amp

    def _calc_cosine_phase(self, state):
        amp = state[..., 0] * state[..., 1]
        displace = -self.dis_factor * state[..., 0] * state[..., 2]
        damping = state[..., 1]
        forcing = self.asymm_forcing
        phase_change = amp + displace - damping + forcing
        return phase_change

    def _calc_sine_phase(self, state):
        amp = state[..., 0] * state[..., 2]
        displace = self.dis_factor * state[..., 0] * state[..., 1]
        damping = state[..., 2]
        phase_change = amp + displace - damping
        return phase_change

    def __call__(self, state):
        westerly_amp = self._calc_westerly(state)
        cosine_change = self._calc_cosine_phase(state)
        sine_change = self._calc_sine_phase(state)

        derivative = torch.stack([westerly_amp, cosine_change, sine_change],
                                 dim=-1)
        return derivative
