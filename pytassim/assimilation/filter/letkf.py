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
from .etkf import ETKFBase
from .etkf_core import ETKFWeightsModule, CorrMixin, UnCorrMixin,\
    ETKFAnalyser

logger = logging.getLogger(__name__)


class _LETKFAnalyser(ETKFAnalyser):
    def __init__(self, localization=None, inf_factor=1.0):
        super().__init__(inf_factor)
        self.localization = localization
        self._gen_weights = ETKFWeightsModule(inf_factor)

    def _localise_obs(self, grid_point, centred_perts, centred_obs, obs_cinv,
                      obs_grid):
        if self.localization is None:
            return centred_perts, centred_obs, obs_cinv
        else:
            use_obs, obs_weights = self.localization.localize_obs(
                grid_point, obs_grid
            )
            obs_weights = torch.as_tensor(obs_weights[use_obs],
                                          dtype=centred_perts.dtype)
            centred_perts = centred_perts[..., use_obs]
            centred_obs = centred_obs[..., use_obs]
            obs_cinv = obs_cinv[use_obs, ...]
            if obs_cinv.dim() == 2:
                obs_cinv = obs_cinv[..., use_obs]
            obs_cinv = obs_cinv * obs_weights
            return centred_perts, centred_obs, obs_cinv


class LETKFBase(ETKFBase):
    def __init__(self, localization=None, inf_factor=1.0, smoother=True,
                 gpu=False, pre_transform=None, post_transform=None):
        self._gen_weights = None
        super().__init__(inf_factor=inf_factor, smoother=smoother, gpu=gpu,
                         pre_transform=pre_transform,
                         post_transform=post_transform)
        self.localization = localization

    @property
    def gen_weights(self):
        return self._gen_weights

    @gen_weights.setter
    def gen_weights(self, new_module):
        if new_module is None:
            self._gen_weights = None
        elif isinstance(new_module, ETKFWeightsModule):
            self._gen_weights = torch.jit.script(new_module)
        else:
            raise TypeError('Given weights module is not a valid '
                            '`ETKFWeightsModule or None!')

    def _localise_obs(self, grid_point, centred_perts, centred_obs, obs_cinv,
                      obs_grid):
        if self.localization is None:
            return centred_perts, centred_obs, obs_cinv
        else:
            use_obs, obs_weights = self.localization.localize_obs(
                grid_point, obs_grid
            )
            obs_weights = torch.as_tensor(obs_weights[use_obs],
                                          dtype=centred_perts.dtype)
            centred_perts = centred_perts[..., use_obs]
            centred_obs = centred_obs[..., use_obs]
            obs_cinv = obs_cinv[use_obs, ...]
            if obs_cinv.dim() == 2:
                obs_cinv = obs_cinv[..., use_obs]
            obs_cinv = obs_cinv * obs_weights
            return centred_perts, centred_obs, obs_cinv

    def _localised_analysis(self, state_perts, centered_perts, centered_obs,
                            obs_cinv, obs_grid):
        grid_first = state_perts.transpose('grid', ...)
        analysis_perts = []
        for sub_perts in grid_first:
            loc_perts, loc_obs, loc_cinv = self._localise_obs(
                sub_perts.grid.values, centered_perts, centered_obs, obs_cinv,
                obs_grid
            )
            analysis_perts.append(loc_ana_perts)
        analysis_perts = xr.concat(analysis_perts, dim='grid')
        analysis_perts = analysis_perts.transpose(*state_perts.dims)
        print(analysis_perts)
        return analysis_perts

    def update_state(self, state, observations, pseudo_state, analysis_time):
        """
        This method updates the state based on given observations and ansub_array.gridalysis
        time. This method prepares different states, localize these states,
        iterates over state grid points, calculates the ensemble  weights and
        applies these weight to localized state. The calculation of the
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
        logger.info('####### Serial LETKF #######')
        logger.info('Starting with applying observation operator')
        pseudo_obs, obs_state, obs_cov, obs_grid = self._get_states(
            pseudo_state, observations,
        )

        logger.info('Scatter the data to torch')
        pseudo_obs, obs_state, obs_cov = self._states_to_torch(
            pseudo_obs, obs_state, obs_cov
        )

        logger.info('Normalise perturbations and observations')
        centred_perts, centred_obs = self._centre_tensors(pseudo_obs, obs_state)
        obs_cinv = self._get_chol_inverse(obs_cov)

        logger.info('Split background into mean and perturbations')
        state_mean, state_perts = state.state.split_mean_perts()

        logger.info('Create analysis perturbations')
        analysis_perts = self._localised_analysis(
            state_perts, centred_perts, centred_obs, obs_cinv, obs_grid
        )
        logger.info('Add background mean to analysis perturbations')
        analysis = (state_mean + analysis_perts).transpose(*state.dims)
        logger.info('Finished with analysis creation')
        return analysis


class LETKFCorr(CorrMixin, LETKFBase):
    """
    This is an implementation of the `localized ensemble transform Kalman
    filter` :cite:`hunt_efficient_2007` for correlated observations.
    This is a localized version of the `ensemble transform Kalman filter`
    :cite:`bishop_adaptive_2001`. This method iterates independently over each
    grid point in given background state. Given localization instance can be
    used to
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
    pass


class LETKFUncorr(UnCorrMixin, LETKFBase):
    """
    This is an implementation of the `localized ensemble transform Kalman
    filter` :cite:`hunt_efficient_2007` for uncorrelated observations.
    This is a localized version of the `ensemble transform Kalman filter`
    :cite:`bishop_adaptive_2001`. This method iterates independently over each
    grid point in given background state. Given localization instance can be
    used to
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
    pass
