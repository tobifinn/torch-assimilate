#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 7/19/19
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
from pytassim.utilities import chol_solve


logger = logging.getLogger(__name__)


def estimate_inc_uncorr(innov, h_jacob, cov_back, cov_obs):
    ht = h_jacob.t()
    bht = torch.mm(cov_back, ht)
    innov_prec = torch.mm(h_jacob, bht)
    if innov_prec.shape[0] > 1:
        mat_size = innov_prec.size()[1]
        step = mat_size + 1
        end = mat_size * mat_size
        innov_prec.view(-1)[:end:step] += cov_obs
        norm_innov = chol_solve(innov_prec, innov).t()
    else:
        innov_prec += cov_obs
        norm_innov = innov / innov_prec
    inc_ana = torch.mm(bht, norm_innov).squeeze(-1)
    return inc_ana


def estimate_inc_corr(innov, h_jacob, cov_back, cov_obs):
    ht = h_jacob.transpose(-1, -2)
    hb = torch.mm(h_jacob, cov_back)
    innov_prec = torch.mm(hb, ht) + cov_obs
    norm_innov = chol_solve(innov_prec, innov).t()
    k_dist = torch.mm(cov_back, ht)
    inc_ana = torch.mm(k_dist, norm_innov).squeeze(-1)
    return inc_ana
