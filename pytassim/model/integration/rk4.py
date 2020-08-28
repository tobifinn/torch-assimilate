#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 11/29/18
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
from typing import Callable, Any

# External modules

# Internal modules
from .integrator import BaseIntegrator


logger = logging.getLogger(__name__)


class RK4Integrator(BaseIntegrator):
    """
    RK4Integrator uses a Runge-Kutta fourth-order method to integrate given
    model function in time. This method calculates four different states, which
    are then averaged to one single state increment. To get another type of
    Runge-Kutta scheme, ``steps`` and ``weights`` can be changed.

    Arguments
    ---------
    model : func
        This model function takes a state and returns a new estimated state. The
        returned state should have the same shape as the input state. The model
        should be a time derivative such that it can be integrated. It is
        assumed that the state does not depend on the time itself.
    dt : float, optional
        This is the integration time step. This time step is unitless and
        depends on model's time unit. A positive time step indicates forward
        integration, while a negative shows a backward integration, which might
        be complicated for given model. Default is 0.05.
    """
    def __init__(self, model: Callable, dt: float = 0.05):
        super().__init__(model=model, dt=dt)
        self.steps = [0, self.dt / 2, self.dt / 2, self.dt]
        self.weights = [1, 2, 2, 1]
        self._weights_sum = sum(self.weights)
        self._weights = [w / self._weights_sum for w in self.weights]

    def __str__(self) -> str:
        return 'RK4Integrator(model={0:s}, dt={1})'.format(str(self.model),
                                                           self.dt)

    def __repr__(self) -> str:
        return 'RK4({0:s})'.format(repr(self.model))

    def _calc_increment(self, state: Any) -> Any:
        """
        This method estimates the increment based estimated slope and set time
        step.

        Parameters
        ----------
        state : any
            This state is used to estimate the slope.

        Returns
        -------
        est_inc : any
            This increment is estimated by multiplying estimated slope with
            set time step.
        """
        est_inc = self._estimate_slope(state) * self.dt
        return est_inc

    def _estimate_slope(self, state: Any) -> Any:
        """
        This method estimates the slope based on given state. This slope is
        used to calculate the increment.

        Parameters
        ----------
        state : any
            This state is used as initial state to estimates the slopes.

        Returns
        -------
        averaged_slope : any
            This slope is averaged based on estimated slopes. These slopes are
            calculated based on given state.
        """
        averaged_slope = state * 0
        curr_slope = state * 0
        for k, ts in enumerate(self.steps):
            model_state = state + curr_slope * ts
            curr_slope = self.model(model_state)
            averaged_slope += self._weights[k] * curr_slope
        return averaged_slope
