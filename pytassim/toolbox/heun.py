#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 2/4/19
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
from torch.optim.optimizer import Optimizer, required

# Internal modules


logger = logging.getLogger(__name__)


class HeunMethod(Optimizer):
    """
    Averaged stochastic gradient descent solve based on Heun's method, known for
    solving ordinary differential equations. This method is based on the
    predictor-corrector method and is almost equal to a runge-kutta second order
    method. If `fast` is selected, the method is based on approximated ``fast``-
    Heun's method.

    This method has been proposed for SGD in `FastHeun: A second-order method
    for stochastic gradient descent`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        fast (bool, optional): If fast Heun should be used (default: True)

    __ ``to be published``
    """
    def __init__(self, params, lr=required, fast=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(HeunMethod, self).__init__(params, defaults)
        self.state['fast'] = fast
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['pred_grad'] = None

    def step(self, closure):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if not self.state['fast'] or state['pred_grad'] is None:
                    if p.grad is None:
                        continue
                    state['pred_grad'] = p.grad.data.clone()
                p.data.add_(-group['lr'], state['pred_grad'])
        loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    continue
                else:
                    p.data.add_(group['lr'], state['pred_grad'])
                    weighted_grad = (state['pred_grad'] + p.grad.data) / 2
                    p.data.add_(-group['lr'], weighted_grad)
                    state['pred_grad'] = p.grad.data.clone()
        return loss
