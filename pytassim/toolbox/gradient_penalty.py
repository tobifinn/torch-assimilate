#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 2/8/19
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
from torch import autograd

# Internal modules


logger = logging.getLogger(__name__)


def zero_grad_penalty(disc_out, disc_input):
    batch_size = disc_input.size(0)
    grad_out = autograd.grad(
        outputs=disc_out.sum(), inputs=disc_input,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_out_squared = grad_out.pow(2)
    penalty = grad_out_squared.view(batch_size, -1).sum(1)
    return penalty
