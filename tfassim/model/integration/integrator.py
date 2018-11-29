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
import abc

# External modules

# Internal modules


logger = logging.getLogger(__name__)


class BaseIntegrator(object):
    """
    This is a base class for integrators. An integrators integrates given model
    function forward or backward in time, depending on given timestep.

    Arguments
    ---------
    model : func
        This model function takes a state and returns a new estimated state. The
        returned state should have the same shape as the input state. The model
        should be a time derivative such that it can be integrated.
    dt : float, optional
        This is the integration time step. This time step is unitless and
        depends on model's time unit. A positive time step indicates forward
        integration, while a negative shows a backward integration, which might
        be complicated for given model. Default is 0.1.
    """

    def __init__(self, model, dt=0.1):
        self._model = None
        self._dt = None
        self.model = model
        self.dt = dt

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        if callable(new_model):
            self._model = new_model
        else:
            raise TypeError('Given model is not callable!')

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, new_dt):
        if not isinstance(new_dt, (float, int)):
            raise TypeError('Given time step is not a float!')
        elif new_dt == 0:
            raise ValueError('Given time step is zero!')
        else:
            self._dt = new_dt

    @abc.abstractmethod
    def _calc_increment(self, state):
        """
        This method estimates the increment based on given state, set model and
        time step.

        Parameters
        ----------
        state : :py:class:``numpy.ndarray``
            This state is used to estimate the increment.

        Returns
        -------
        est_inc : :py:class:``numpy.ndarray``
            This increment is estimated by this integration object.
        """
        pass

    def integrate(self, state):
        """
        This method integrates given model by set time step. Given state is used
        as initial state and passed to model.

        Parameters
        ----------
        state : :py:class:``numpy.ndarray``
            This state is used as initial state for the integration. This state
            is passed to set model.

        Returns
        -------
        int_state : :py:class:``numpy.ndarray``
            This state is integrated by given model. The integrated state is the
            initial state plus an increment estimated based on this integrator
            and set model.
        """
        estimated_inc = self._calc_increment(state)
        int_state = state + estimated_inc
        return int_state
