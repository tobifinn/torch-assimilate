#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 10.08.20
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}
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


def evd(tensor, reg_value=0):
    evals, evects = torch.symeig(tensor, eigenvectors=True, upper=False)
    evals = evals.clamp(min=0)
    evals = evals + reg_value
    evals_inv = 1 / evals
    evects_inv = evects.t()
    return evals, evects, evals_inv, evects_inv


def rev_evd(evals, evects, evects_inv):
    diag_flat_evals = torch.diagflat(evals)
    rev_evd = torch.mm(evects, diag_flat_evals)
    rev_evd = torch.mm(rev_evd, evects_inv)
    return rev_evd
