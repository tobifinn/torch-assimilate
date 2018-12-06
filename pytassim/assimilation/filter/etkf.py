#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12/6/18
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
import xarray as xr
import torch

# Internal modules
import pytassim.state
from .filter import FilterAssimilation

logger = logging.getLogger(__name__)


class ETKFilter(FilterAssimilation):
    def update_state(self, state, observations, analysis_time):
        prepared_states = self._prepare(
            state, observations
        )[:-1]
        torch_states = [torch.tensor(s) for s in prepared_states]
        w_mean, w_pert = self._gen_weights(*torch_states)

    def get_weights(self, state, observations):
        pass

    def _prepare(self, state, observations):
        hx_mean, hx_pert, filtered_obs = self._prepare_back_obs(state,
                                                                observations)
        obs_state, obs_cov, obs_grid = self._prepare_obs(filtered_obs)
        return hx_mean, hx_pert, obs_state, obs_cov, obs_grid

    def _prepare_back_obs(self, state, observations):
        pseudo_obs, filtered_obs = self._apply_obs_operator(state, observations)
        pseudo_obs = [obs.stack(obs_id=('time', 'obs_grid_1'))
                      for obs in pseudo_obs]
        pseudo_obs_concat = xr.concat(pseudo_obs, dim='obs_id')
        hx_mean, hx_pert = pseudo_obs_concat.state.split_mean_perts()
        return hx_mean.values, hx_pert.T.values, filtered_obs

    def _compute_c(self, hx_pert, obs_cov, obs_weights=1):
        pinv = torch.pinverse(obs_cov)
        calculated_c = torch.matmul(pinv, hx_pert).t()
        calculated_c = calculated_c * obs_weights
        return calculated_c

    def _calc_precision(self, c, hx_pert):
        ens_size = hx_pert.size()[1]
        prec_obs = torch.matmul(c, hx_pert)
        prec_back = (ens_size - 1) * torch.eye(ens_size).double()
        prec_ana = prec_back + prec_obs
        return prec_ana

    def _det_square_root(self, evals_inv, evects):
        ens_size = evals_inv.size()[0]
        w_perts = torch.sqrt((ens_size - 1) * evals_inv)
        w_perts = torch.matmul(evects.t(), w_perts)
        w_perts = torch.matmul(w_perts, evects)
        return w_perts

    @staticmethod
    def _eigendecomp(precision):
        evals, evects = torch.eig(precision, eigenvectors=True)
        evals = evals[:, 0]
        evals_inv = 1 / evals
        return evals, evects, evals_inv

    def _gen_weights(self, hx_mean, hx_pert, obs_state, obs_cov, obs_weights=1):
        estimated_c = self._compute_c(hx_pert, obs_cov, obs_weights)
        prec_ana = self._calc_precision(estimated_c, hx_pert)
        evals, evects, evals_inv = self._eigendecomp(prec_ana)

        cov_analysed = torch.matmul(evects.t(), torch.diagflat(evals_inv))
        cov_analysed = torch.matmul(cov_analysed, evects)
        innov = obs_state - hx_mean
        gain = torch.matmul(cov_analysed, estimated_c)
        w_mean = torch.matmul(gain, innov)

        w_perts = self._det_square_root(evals_inv, evects)
        return w_mean, w_perts

    def _apply_weights(self, w_mean, w_pert, state_mean, state_pert):
        pass
