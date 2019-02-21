#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12/7/18
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
from tqdm import tqdm
import torch

# Internal modules
from .etkf import ETKFCorr
from . import etkf_core

logger = logging.getLogger(__name__)


def local_etkf(ind, innov, hx_perts, obs_cov, back_prec, obs_grid, state_grid,
               state_perts, localization=None):
    obs_weights = 1
    if localization:
        use_obs, obs_weights = localization.localize_obs(
            state_grid[ind], obs_grid
        )
        obs_weights = torch.as_tensor(obs_weights[use_obs], dtype=innov.dtype)
        use_obs = torch.ByteTensor(use_obs.astype(int))
        if innov.is_cuda:
            obs_weights = obs_weights.cuda()
        innov = innov[use_obs]
        hx_perts = hx_perts[use_obs]
        obs_cov = obs_cov[use_obs, :][:, use_obs]
    w_mean_l, w_perts_l = etkf_core.gen_weights_corr(
        back_prec, innov, hx_perts, obs_cov, obs_weights
    )
    weights_l = (w_mean_l+w_perts_l).t()
    ana_state_l = torch.matmul(state_perts[ind], weights_l)
    return ana_state_l, weights_l, w_mean_l


class LETKFilter(ETKFCorr):
    """
    This is an implementation of the `localized ensemble transform Kalman
    filter` :cite:`hunt_efficient_2007`, which is a localized version of the
    `ensemble transform Kalman filter` :cite:`bishop_adaptive_2001`. This method
    iterates independently over each grid
    point in given background state. Given localization instance can be used to
    constrain the influence of observations in space. The ensemble weights are
    calculated for every grid point and independently applied to every grid
    point. This implementation follows :cite:`hunt_efficient_2007`, with local
    weight estimation and is implemented in PyTorch. This implementation allows
    filtering in time based on linear propagation assumption
    :cite:`hunt_four-dimensional_2004` and ensemble smoothing.

    Parameters
    ----------
    smoothing : bool, optional
        Indicates if this filter should be run in smoothing or in filtering
        mode. In smoothing mode, no analysis time is selected from given state
        and the ensemble weights are applied to the whole state. In filtering
        mode, the weights are applied only on selected analysis time. Default
        is False, indicating filtering mode.
    localization : obj or None, optional
        This localization is used to localize and constrain observations
        spatially. If this localization is None, no localization is applied such
        it is an inefficient version of the `ensemble transform Kalman filter`.
        Default value is None, indicating no localization at all.
    inf_factor : float, optional
        Multiplicative inflation factor :math:`\\rho``, which is applied to the
        background precision. An inflation factor greater one increases the
        ensemble spread, while a factor less one decreases the spread. Default
        is 1.0, which is the same as no inflation at all.
    gpu : bool, optional
        Indicator if the weight estimation should be done on either GPU (True)
        or CPU (False): Default is None. For small models, estimation of the
        weights on CPU is faster than on GPU!.
    """

    def __init__(self, localization=None, inf_factor=1.0, smoother=True,
                 gpu=False, pre_transform=None, post_transform=None):
        super().__init__(inf_factor=inf_factor, smoother=smoother, gpu=gpu,
                         pre_transform=pre_transform,
                         post_transform=post_transform)
        self.localization = localization

    def update_state(self, state, observations, analysis_time):
        """
        This method updates the state based on given observations and analysis
        time. This method prepares different states, localize these states,
        iterates over state grid points, calculates the ensemble  weights and
        applies these weight to localized state. The calculation of the
        weights is based on PyTorch, while everything else is calculated with
        Numpy / Xarray.

        Parameters
        ----------
        state : :py:class:`xarray.DataArray`
            This state is used to generate an observation-equivalent. It is
            further updated by this assimilation algorithm and given
            ``observation``. This :py:class:`~xarray.DataArray` should have
            four coordinates, which are specified in
            :py:class:`pytassim.state.ModelState`.
        observations : :py:class:`xarray.Dataset` or \
        iterable(:py:class:`xarray.Dataset`)
            These observations are used to update given state. An iterable of
            many :py:class:`xarray.Dataset` can be used to assimilate different
            variables. For the observation state, these observations are
            stacked such that the observation state contains all observations.
        analysis_time : :py:class:`datetime.datetime`
            This analysis time determines at which point the state is updated.

        Returns
        -------
        analysis : :py:class:`xarray.DataArray`
            The analysed state based on given state and observations. The
            analysis has same coordinates as given ``state``. If filtering mode
            is on, then the time axis has only one element.
        """
        logger.info('####### Serial LETKF #######')
        logger.info('Starting with specific preparation')
        innov, hx_perts, obs_cov, obs_grid = self._prepare(state, observations)
        back_state = state.transpose('grid', 'var_name', 'time', 'ensemble')
        state_mean, state_perts = back_state.state.split_mean_perts()

        logger.info('Transfering the data to torch')
        back_prec = self._get_back_prec(len(back_state.ensemble))
        innov, hx_perts, obs_cov, back_state = self._states_to_torch(
            innov, hx_perts, obs_cov, state_perts.values,
        )
        state_grid = state_perts.grid.values
        len_state_grid = len(state_grid)
        grid_inds = range(len_state_grid)

        delta_ana = []
        weights = []
        logger.info('Iterating through state grid')
        for grid_ind in tqdm(grid_inds, total=len_state_grid):
            ana_l, weights_l, _ = local_etkf(
                grid_ind, innov, hx_perts, obs_cov, back_prec, obs_grid,
                state_grid, back_state, self.localization
            )
            delta_ana.append(ana_l)
            weights.append(weights_l)
        delta_ana = torch.stack(delta_ana, dim=0)
        state_perts.values = delta_ana.numpy()
        weights = torch.stack(weights, dim=0).numpy()
        self._weights = self._get_weight_array(
            weights, grid=state_grid, ensemble=state.ensemble.values
        )
        analysis = (state_mean + state_perts).transpose(*state.dims)
        logger.info('Finished with analysis creation')
        return analysis

    @staticmethod
    def _get_weight_array(weights, grid, ensemble):
        weights_da = xr.DataArray(
            weights,
            coords={
                'grid': grid,
                'ensemble_1': ensemble,
                'ensemble_2': ensemble
            },
            dims=['grid', 'ensemble_1', 'ensemble_2']
        )
        return weights_da

    def _localize(self, grid_ind, prepared_states):
        try:
            use_obs, obs_weights = self.localization.localize_obs(
                grid_ind, prepared_states[-1]
            )
            innov = prepared_states[0][use_obs]
            hx_perts = prepared_states[1][use_obs]
            obs_cov = prepared_states[2][use_obs, :][:, use_obs]
            obs_weights = obs_weights[use_obs]
            return innov, hx_perts, obs_cov, obs_weights
        except (NotImplementedError, AttributeError):
            return prepared_states[:-1]
