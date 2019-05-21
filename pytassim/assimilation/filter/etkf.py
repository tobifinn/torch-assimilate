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

import numpy as np
import pandas as pd
import torch
# External modules
import xarray as xr

from .etkf_core import gen_weights_corr, gen_weights_uncorr
# Internal modules
from .filter import FilterAssimilation

logger = logging.getLogger(__name__)


class ETKFCorr(FilterAssimilation):
    """
    This is an implementation of the `ensemble transform Kalman filter`
    :cite:`bishop_adaptive_2001` for correlated observation covariances.
    This ensemble Kalman filter is a deterministic filter, where the state is
    update globally. This ensemble Kalman filter estimates ensemble weights in
    weight space, which are then applied to the given state. This implementation
    follows :cite:`hunt_efficient_2007` with global weight estimation and is
    implemented in PyTorch.
    This implementation allows filtering in time based on linear propagation
    assumption :cite:`hunt_four-dimensional_2004` and ensemble smoothing.

    Parameters
    ----------
    smoothing : bool, optional
        Indicates if this filter should be run in smoothing or in filtering
        mode. In smoothing mode, no analysis time is selected from given state
        and the ensemble weights are applied to the whole state. In filtering
        mode, the weights are applied only on selected analysis time. Default
        is False, indicating filtering mode.
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
    def __init__(self, inf_factor=1.0, smoother=True, gpu=False,
                 pre_transform=None, post_transform=None):
        super().__init__(smoother=smoother, gpu=gpu,
                         pre_transform=pre_transform,
                         post_transform=post_transform)
        self.inf_factor = inf_factor
        self._back_prec = None
        self._weights = None
        self._gen_weights_func = gen_weights_corr

    @property
    def weights(self):
        return self._weights

    def update_state(self, state, observations, pseudo_state, analysis_time):
        """
        This method updates the state based on given observations and analysis
        time. This method prepares the different states, calculates the ensemble
        weights and applies these weight to given state. The calculation of the
        weights is based on PyTorch, while everything else is calculated with
        Numpy / Xarray.

        Parameters
        ----------
        state : :py:class:`xarray.DataArray`
            This state is updated by this assimilation algorithm and given
            ``observation``. This :py:class:`~xarray.DataArray` should have
            four coordinates, which are specified in
            :py:class:`pytassim.state.ModelState`.
        observations : :py:class:`xarray.Dataset` or \
        iterable(:py:class:`xarray.Dataset`)
            These observations are used to update given state. An iterable of
            many :py:class:`xarray.Dataset` can be used to assimilate different
            variables. For the observation state, these observations are
            stacked such that the observation state contains all observations.
        pseudo_state : :py:class:`xarray.DataArray`
            This state is used to generate an observation-equivalent. This
             :py:class:`~xarray.DataArray` should have four coordinates, which
             are specified in :py:class:`pytassim.state.ModelState`.
        analysis_time : :py:class:`datetime.datetime`
            This analysis time determines at which point the state is updated.

        Returns
        -------
        analysis : :py:class:`xarray.DataArray`
            The analysed state based on given state and observations. The
            analysis has same coordinates as given ``state``. If filtering mode
            is on, then the time axis has only one element.
        """
        logger.info('####### Global ETKF #######')
        logger.info('Starting with specific preparation')
        prepared_states = self._prepare(
            pseudo_state, observations,
        )[:-1]
        logger.info('Transfering the data to torch')
        innov, hx_perts, obs_cov = self._states_to_torch(*prepared_states)
        back_prec = self._get_back_prec(len(state.ensemble))
        logger.info('Gathering the weights')
        w_mean, w_perts = self._gen_weights_func(
            back_prec, innov, hx_perts, obs_cov
        )
        logger.info('Applying weights to state')
        state_mean, state_perts = state.state.split_mean_perts()
        analysis = self._apply_weights(w_mean, w_perts, state_mean, state_perts)
        analysis = analysis.transpose('var_name', 'time', 'ensemble', 'grid')
        logger.info('Finished with analysis creation')
        return analysis

    def _prepare(self, pseudo_state, observations):
        """
        This method prepares the different parts of the state. It calculates
        statistics in observation space and concatenates given observations into
        a long vector. Observations without an observation operator, defined in
        :py:meth:`xarray.Dataset.obs.operator`, are skipped. This method
        prepares the states in Numpy / Xarray.

        Parameters
        ----------
        pseudo_state : :py:class:`xarray.DataArray`
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

        Returns
        -------
        innov : :py:class:`numpy.ndarray`
            This vector contains the innovations, calculated with
            :math:`\\textbf{y}^{o} - \\overline{h(\\textbf{x}^{b})}`. The length
            of this vector is :math:`l`, the observation length.
        hx_perts : :py:class:`numpy.ndarray`
            This matrix contains the ensemble perturbations in ensemble space.
            The `i`-th perturbation is calculated based on
            :math:`h(\\textbf{x}_{i}^{b})-\\overline{h(\\textbf{x}^{b})}`. The
            shape of this matrix is :math:`l~x~k`, with :math:`k` as ensemble
            size and :math:`l` as observation length.
        obs_cov : :py:class:`numpy.ndarray`
            The concatenated observation covariance. This covariance is created
            with the assumption that different observation subset are not
            correlated. This matrix has a shape of :math:`l~x~l`, with :math:`l`
            as observation length.
        obs_grid : :py:class:`numpy.ndarray`
            This is the concatenated observation grid. This can be used for
            localization or weighting purpose. This last axis of this array has
            a length of :math:`l`, the observation length.
        """
        logger.info('Apply observation operator')
        hx_mean, hx_perts, filtered_obs = self._prepare_back_obs(pseudo_state,
                                                                 observations)
        logger.info('Concatenate observations')
        obs_state, obs_cov, obs_grid = self._prepare_obs(filtered_obs)
        innov = obs_state - hx_mean
        return innov, hx_perts, obs_cov, obs_grid

    def _get_back_prec(self, ens_mems):
        back_prec = torch.eye(ens_mems, dtype=self.dtype)
        if self.gpu:
            back_prec = back_prec.cuda()
        back_prec *= (ens_mems - 1)
        back_prec /= self.inf_factor
        return back_prec

    def _prepare_back_obs(self, state, observations):
        pseudo_obs, filtered_obs = self._apply_obs_operator(state, observations)
        hx_mean, hx_perts = self._prepare_pseudo_obs(pseudo_obs)
        return hx_mean, hx_perts, filtered_obs

    @staticmethod
    def _prepare_pseudo_obs(pseudo_obs):
        state_stacked_list = []
        for obs in pseudo_obs:
            if isinstance(obs.indexes['obs_grid_1'], pd.MultiIndex):
                obs['obs_grid_1'] = pd.Index(
                    obs.indexes['obs_grid_1'].values, tupleize_cols=False
                )
            stacked_obs = obs.stack(obs_id=('time', 'obs_grid_1'))
            state_stacked_list.append(stacked_obs)
        pseudo_obs_concat = xr.concat(state_stacked_list, dim='obs_id')
        hx_mean, hx_perts = pseudo_obs_concat.state.split_mean_perts()
        return hx_mean.values, hx_perts.T.values

    @staticmethod
    def _weights_matmul(perts, weights):
        ana_perts = xr.apply_ufunc(
            np.matmul, perts, weights,
            input_core_dims=[['ensemble'], []], output_core_dims=[['ensemble']],
            dask='parallelized'
        )
        return ana_perts

    def _apply_weights(self, w_mean, w_perts, state_mean, state_pert):
        """
        This method applies given weights to given state. The weights are
        combined column-wise for efficient computing and applied to given
        state perturbations. These non-centered analysed perturbations are then
        added to given state mean to get the analysis.

        Parameters
        ----------
        w_mean : :py:class:`torch.tensor`
            The estimated ensemble mean weights. These weights are column-wise
            added to the weight perturbations. The shape of this tensor is
            :math:`k`, the ensemble size.
        w_perts : :py:class:`torch.tensor`
            The estimated ensemble perturbations in weight space. These weights
            are used to estimate new centered ensemble perturbations. The
            shape of this tensor is :math:`k~x~k`, with :math:`k` as ensemble
            size.
        state_mean : :py:class:`xarray.DataArray`
            This is the state mean, which is updated by non-centered analysed
            perturbations.
        state_pert : :py:class:`xarray.DataArray`
            These ensemble perturbations are used to estimate new non-centered
            analysed perturbations.

        Returns
        -------
        analysis : :py:class:`xarray.DataArray`
            The estimated analysis based on given state and weights.
        """
        combined_weights = (w_mean + w_perts).t()
        if self.gpu:
            combined_weights = combined_weights.cpu()
        ana_perts = self._weights_matmul(
            state_pert, combined_weights.numpy()
        )
        analysis = state_mean + ana_perts
        return analysis


class ETKFUncorr(ETKFCorr):
    """
    This is an implementation of the `ensemble transform Kalman filter`
    :cite:`bishop_adaptive_2001` for uncorrelated observation covariances.
    This ensemble Kalman filter is a deterministic filter, where the state is
    update globally. This ensemble Kalman filter estimates ensemble weights in
    weight space, which are then applied to the given state. This implementation
    follows :cite:`hunt_efficient_2007` with global weight estimation and is
    implemented in PyTorch.
    This implementation allows filtering in time based on linear propagation
    assumption :cite:`hunt_four-dimensional_2004` and ensemble smoothing.

    Parameters
    ----------
    smoothing : bool, optional
        Indicates if this filter should be run in smoothing or in filtering
        mode. In smoothing mode, no analysis time is selected from given state
        and the ensemble weights are applied to the whole state. In filtering
        mode, the weights are applied only on selected analysis time. Default
        is False, indicating filtering mode.
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
    def __init__(self, inf_factor=1.0, smoother=True, gpu=False,
                 pre_transform=None, post_transform=None):
        super().__init__(inf_factor=inf_factor, smoother=smoother, gpu=gpu,
                         pre_transform=pre_transform,
                         post_transform=post_transform)
        self._gen_weights_func = gen_weights_uncorr
        self._correlated = False
