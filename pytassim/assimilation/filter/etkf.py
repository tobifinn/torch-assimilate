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
import numpy as np

# Internal modules
import pytassim.state
from .filter import FilterAssimilation

logger = logging.getLogger(__name__)


class ETKFilter(FilterAssimilation):
    """
    This is an implementation of the `ensemble transform Kalman filter` [B01]_.
    This ensemble Kalman filter is a deterministic filter, where the state is
    update globally. This ensemble Kalman filter estimates ensemble weights in
    weight space, which are then applied to the given state. This implementation
    follows [H07]_ with global weight estimation and is implemented in PyTorch.
    This implementation allows filtering in time based on linear propagation
    assumption [H04]_ and ensemble smoothing.

    References
    ----------
    .. [B01] Bishop, C. H., Etherton, B. J., & Majumdar, S. J. (2001).
             Adaptive sampling with the ensemble transform Kalman filter.
             Part I: Theoretical aspects. Monthly Weather Review, 129(3),
             420–436.
    .. [H04] Hunt, B., et al. Four-dimensional ensemble Kalman filtering.
             Tellus A, 56(4), 273–277.
    .. [H07] Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007).
             Efficient data assimilation for spatiotemporal chaos: A local
             ensemble transform Kalman filter. Physica D: Nonlinear
             Phenomena, 230(1), 112–126.
             https://doi.org/10.1016/j.physd.2006.11.008

    Parameters
    ----------
    smoothing : bool, optional
        Indicates if this filter should be run in smoothing or in filtering
        mode. In smoothing mode, no analysis time is selected from given state
        and the ensemble weights are applied to the whole state. In filtering
        mode, the weights are applied only on selected analysis time. Default
        is False, indicating filtering mode.
    """
    def __init__(self, smoothing=False):
        super().__init__()
        self.smoothing = smoothing

    def update_state(self, state, observations, analysis_time):
        """
        This method updates the state based on given observations and analysis
        time. This method prepares the different states, calculates the ensemble
        weights and applies these weight to given state. The calculation of the
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
        prepared_states = self._prepare(
            state, observations
        )[:-1]
        torch_states = [torch.tensor(s) for s in prepared_states]
        w_mean, w_perts = self._gen_weights(*torch_states)
        if not self.smoothing:
            analysis_state = state.sel(time=[analysis_time, ])
        else:
            analysis_state = state
        state_mean, state_perts = analysis_state.state.split_mean_perts()
        analysis = self._apply_weights(w_mean, w_perts, state_mean, state_perts)
        return analysis

    def _prepare(self, state, observations):
        """
        This method prepares the different parts of the state. It calculates
        statistics in observation space and concatenates given observations into
        a long vector. Observations without an observation operator, defined in
        :py:meth:`xarray.Dataset.obs.operator`, are skipped. This method
        prepares the states in Numpy / Xarray.

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
        hx_mean, hx_perts, filtered_obs = self._prepare_back_obs(state,
                                                                 observations)
        obs_state, obs_cov, obs_grid = self._prepare_obs(filtered_obs)
        innov = obs_state - hx_mean
        return innov, hx_perts, obs_cov, obs_grid

    def _prepare_back_obs(self, state, observations):
        pseudo_obs, filtered_obs = self._apply_obs_operator(state, observations)
        pseudo_obs = [obs.stack(obs_id=('time', 'obs_grid_1'))
                      for obs in pseudo_obs]
        pseudo_obs_concat = xr.concat(pseudo_obs, dim='obs_id')
        hx_mean, hx_perts = pseudo_obs_concat.state.split_mean_perts()
        return hx_mean.values, hx_perts.T.values, filtered_obs

    @staticmethod
    def _compute_c(hx_perts, obs_cov, obs_weights=1):
        pinv = torch.pinverse(obs_cov)
        calculated_c = torch.matmul(pinv, hx_perts).t()
        calculated_c = calculated_c * obs_weights
        return calculated_c

    @staticmethod
    def _calc_precision(c, hx_perts):
        ens_size = hx_perts.size()[1]
        prec_obs = torch.matmul(c, hx_perts)
        prec_back = (ens_size - 1) * torch.eye(ens_size).double()
        prec_ana = prec_back + prec_obs
        return prec_ana

    @staticmethod
    def _det_square_root(evals_inv, evects):
        ens_size = evals_inv.size()[0]
        w_perts = torch.sqrt((ens_size - 1) * evals_inv)
        w_perts = torch.matmul(evects.t(), torch.diagflat(w_perts))
        w_perts = torch.matmul(w_perts, evects)
        return w_perts

    @staticmethod
    def _eigendecomp(precision):
        evals, evects = torch.eig(precision, eigenvectors=True)
        evals = evals[:, 0]
        evals_inv = 1 / evals
        return evals, evects, evals_inv

    def _gen_weights(self, innov, hx_perts, obs_cov, obs_weights=1):
        """
        This method is the main method to calculates the ensemble weights,
        based on [H07]_. To generate the weights, the given arguments have to be
        prepared and in a special format. The weights are estimated with
        PyTorch.

        Parameters
        ----------
        innov : :py:class:`torch.tensor`
            These innovations are multiplied by the ensemble gain to estimate
            the mean ensemble weights. These innovation should have a shape of
            :math:`l`, the observation length.
        hx_perts : :py:class:`torch.tensor`
            These are the ensemble perturbations in ensemble space. These
            perturbations are used to calculated the analysed ensemble
            covariance in weight space. These perturbations have a shape of
            :math:`l~x~k`, with :math:`k` as ensemble size and :math:`l` as
            observation length.
        obs_cov : :py:class:`torch.tensor`
            This tensor represents the observation covariance. This covariance
            is used for the estimation of the analysed covariance in weight
            space. The shape of this covariance should be :math:`l~x~l`, with
            :math:`l` as observation length.
        obs_weights : :py:class:`torch.tensor` or float, optional
            These are the observation weights. These observation weights can be
            used for localization or weighting purpose. If these observation
            weights are a float, then the same weight for every observation is
            used. If these weights are a :py:class:`~torch.tensor`, then the
            shape of this tensor should be :math:`l`, the observation length.
            The default values is 1, indicating that every observation is
            uniformly weighted.

        Returns
        -------
        w_mean : :py:class:`torch.tensor`
            The estimated ensemble mean weights. These weights can be used to
            update the ensemble mean. The shape of this tensor is :math:`k`, the
            ensemble size.
        w_perts : :py:class:`torch.tensor`
            The estimated ensemble perturbations in weight space. These weights
            can be used to estimate new centered ensemble perturbations. The
            shape of this tensor is :math:`k~x~k`, with :math:`k` as ensemble
            size.

        References
        ----------
        .. [H07] Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007).
                 Efficient data assimilation for spatiotemporal chaos: A local
                 ensemble transform Kalman filter. Physica D: Nonlinear
                 Phenomena, 230(1), 112–126.
                 https://doi.org/10.1016/j.physd.2006.11.008
        """
        estimated_c = self._compute_c(hx_perts, obs_cov, obs_weights)
        prec_ana = self._calc_precision(estimated_c, hx_perts)
        evals, evects, evals_inv = self._eigendecomp(prec_ana)

        cov_analysed = torch.matmul(evects.t(), torch.diagflat(evals_inv))
        cov_analysed = torch.matmul(cov_analysed, evects)
        gain = torch.matmul(cov_analysed, estimated_c)
        w_mean = torch.matmul(gain, innov)

        w_perts = self._det_square_root(evals_inv, evects)
        return w_mean, w_perts

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
        combined_weights = (w_mean+w_perts).numpy()
        ana_perts = self._weights_matmul(state_pert, combined_weights)
        analysis = state_mean + ana_perts
        analysis = analysis.transpose('var_name', 'time', 'ensemble', 'grid')
        return analysis
