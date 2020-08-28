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
import abc
from typing import Union, Iterable, Tuple, List

# External modules
import xarray as xr
import numpy as np
import pandas as pd
import torch

# Internal modules
from .etkf_core import ETKFAnalyser, CorrMixin, UnCorrMixin
from .filter import FilterAssimilation
from pytassim.transform import BaseTransformer


logger = logging.getLogger(__name__)


__all__ = [
    'ETKFCorr',
    'ETKFUncorr'
]


class ETKFBase(FilterAssimilation):
    """
    The base object for the ensemble transform Kalman filter.
    """
    def __init__(
            self,
            inf_factor: Union[float, torch.Tensor, torch.nn.Parameter] = 1.0,
            smoother: bool = False, gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None
    ):
        super().__init__(smoother=smoother, gpu=gpu,
                         pre_transform=pre_transform,
                         post_transform=post_transform)
        self._name = 'Global ETKF'
        self._weights = None
        self.inf_factor = inf_factor

    def __str__(self):
        return 'Global ETKF({0})'.format(self.inf_factor)

    def __repr__(self):
        return 'ETKF'

    @property
    def inf_factor(self) -> Union[float, torch.Tensor, torch.nn.Parameter]:
        return self._analyser.inf_factor

    @inf_factor.setter
    def inf_factor(
            self,
            new_factor: Union[float, torch.Tensor, torch.nn.Parameter]
    ):
        """
        Sets a new inflation factor.
        """
        self._analyser = ETKFAnalyser(inf_factor=new_factor)

    @property
    def analyser(self) -> ETKFAnalyser:
        return self._analyser

    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    def _normalise_obs(
            self,
            pseudo_obs: torch.Tensor,
            obs: torch.Tensor,
            cinv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalise given pseudo observations and observations with given
        inverse cholesky decomposition of the observational covariance.
        """
        pseudo_mean = pseudo_obs.mean(dim=-2, keepdim=True)
        normed_perts = self._mul_cinv(pseudo_obs-pseudo_mean, cinv)
        normed_obs = self._mul_cinv(obs.view(1, -1)-pseudo_mean, cinv)
        return normed_perts, normed_obs

    @abc.abstractmethod
    def _mul_cinv(
            self,
            state: torch.Tensor,
            cinv: torch.Tensor
    ) -> torch.Tensor:
        """
        Multiply given state with given inverse cholesky decomposition of the
        observational covariance.
        """
        pass

    @abc.abstractmethod
    def _get_chol_inverse(self, cov: torch.Tensor) -> torch.Tensor:
        """
        Estimate the cholesky inverse of given covariance matrix.
        """
        pass

    def update_state(
            self,
            state: xr.DataArray,
            observations: Union[xr.Dataset, Iterable[xr.Dataset]],
            pseudo_state: xr.DataArray,
            analysis_time: pd.Timestamp
    ) -> xr.DataArray:
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
        logger.info('####### {0:s} #######'.format(self._name))
        logger.info('Starting with specific preparation')
        pseudo_obs, obs_state, obs_cov, obs_grid = self._get_states(
            pseudo_state, observations,
        )

        logger.info('Transfering the data to torch')
        pseudo_obs, obs_state, obs_cov = self._states_to_torch(
            pseudo_obs, obs_state, obs_cov
        )

        logger.info('Normalise perturbations and observations')
        obs_cinv = self._get_chol_inverse(obs_cov)
        normed_perts, normed_obs = self._normalise_obs(pseudo_obs, obs_state,
                                                       obs_cinv)
        state_mean, state_perts = state.state.split_mean_perts()
        state_grid = state_perts.indexes['grid']
        state_perts, = self._states_to_torch(state_perts.values)

        logger.info('Create analysis perturbations')
        analysis_perts = self.analyser(state_perts, normed_perts, normed_obs,
                                       state_grid, obs_grid)

        logger.info('Create analysis')
        analysis_perts = state.copy(data=analysis_perts.numpy())
        analysis = analysis_perts + state_mean
        analysis = analysis.transpose('var_name', 'time', 'ensemble', 'grid')
        return analysis

    def _get_states(
            self,
            pseudo_state: xr.DataArray,
            observations: Union[xr.Dataset, Iterable[xr.Dataset]]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        pseudo_obs, filtered_obs = self._get_pseudo_obs(pseudo_state,
                                                        observations)
        logger.info('Concatenate observations')
        obs_state, obs_grid = self._prepare_obs(filtered_obs)
        obs_cov = self._get_obs_cov(filtered_obs)
        return pseudo_obs, obs_state, obs_cov, obs_grid

    def _get_pseudo_obs(
            self,
            state: xr.DataArray,
            observations: Union[xr.Dataset, Iterable[xr.Dataset]]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Get pseudo observational array and filtered observations. This method
        applies the observation operator and concatenates the pseudo
        observations.
        """
        pseudo_obs, filtered_obs = self._apply_obs_operator(state, observations)
        pseudo_obs = self._cat_pseudo_obs(pseudo_obs)
        return pseudo_obs, filtered_obs

    @staticmethod
    def _cat_pseudo_obs(pseudo_obs: Iterable[xr.DataArray]) -> np.ndarray:
        """
        Concatenate given pseudo observations into a pseudo observational
        array.
        """
        state_stacked_list = []
        for obs in pseudo_obs:
            if isinstance(obs.indexes['obs_grid_1'], pd.MultiIndex):
                obs['obs_grid_1'] = pd.Index(
                    obs.indexes['obs_grid_1'].values, tupleize_cols=False
                )
            stacked_obs = obs.stack(obs_id=('time', 'obs_grid_1'))
            state_stacked_list.append(stacked_obs)
        pseudo_obs_concat = xr.concat(state_stacked_list, dim='obs_id')
        pseudo_obs_concat = pseudo_obs_concat.data
        return pseudo_obs_concat


class ETKFCorr(CorrMixin, ETKFBase):
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
    smoother : bool, optional
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
    def __str__(self) -> str:
        return 'Correlated {0:s}'.format(str(super(ETKFBase)))

    def __repr__(self) -> str:
        return 'Corr{0:s}'.format(repr(super(ETKFBase)))


class ETKFUncorr(UnCorrMixin, ETKFBase):
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
    smoother : bool, optional
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
    def __str__(self) -> str:
        return 'Uncorrelated {0:s}'.format(str(super(ETKFBase)))

    def __repr__(self) -> str:
        return 'Uncorr{0:s}'.format(repr(super(ETKFBase)))
