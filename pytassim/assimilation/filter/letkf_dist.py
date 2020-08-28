#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 2/14/19
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
import operator
from typing import Union, Type, Iterable

# External modules
import numpy as np
import torch
import xarray as xr
import pandas as pd

import dask
import dask.array as da
from distributed import Client
from distributed.deploy.cluster import Cluster

# Internal modules
from .etkf_core import CorrMixin, UnCorrMixin
from .letkf import LETKFBase

from pytassim.localization import BaseLocalization
from pytassim.transform import BaseTransformer


logger = logging.getLogger(__name__)


class DistributedLETKFBase(LETKFBase):
    """
    Base object for a distributed localised ensemble transform Kalman filter.
    """
    def __init__(
            self,
            client: Union[None, Client] = None,
            cluster: Union[None, Type[Cluster]] = None,
            chunksize: int = 10,
            localization: Union[None, BaseLocalization] = None,
            inf_factor: Union[torch.Tensor, float, torch.nn.Parameter] = 1.0,
            smoother: bool = False, gpu: bool = False,
            pre_transform: Union[None, Iterable[Type[BaseTransformer]]] = None,
            post_transform: Union[None, Iterable[Type[BaseTransformer]]] = None
    ):
        super().__init__(localization, inf_factor, smoother, gpu, pre_transform,
                         post_transform)
        self._name = 'Distributed LETKF'
        self._cluster = None
        self._client = None
        self._chunksize = 1
        self.chunksize = chunksize
        self.set_client_cluster(client=client, cluster=cluster)

    @staticmethod
    def _validate_client(client) -> bool:
        return isinstance(client, Client)

    @staticmethod
    def _validate_cluster(cluster) -> bool:
        return hasattr(cluster, "scheduler_address")

    def _check_client_cluster(
            self,
            client: Union[None, Client],
            cluster: Union[None, Type[Cluster]],
    ):
        not_valid_cluster = not self._validate_cluster(cluster)
        not_valid_client = not self._validate_client(client)
        if not_valid_client and not_valid_cluster:
            raise ValueError(
                'Either a client or a cluster have to be specified!'
            )

    @property
    def cluster(self) -> Type[Cluster]:
        return self._cluster

    @property
    def client(self) -> Client:
        return self._client

    def set_client_cluster(
            self,
            client: Union[None, Client] = None,
            cluster: Union[None, Type[Cluster]] = None,
    ):
        """
        This method sets the client and cluster. If both are given and valid,
        then client has priority.

        Parameters
        ----------
        client : :py:class:``~dask.distributed.Client`` or None
            This dask distributed client is used to parallelize the processes.
            Either this client or ``cluster`` has to be
            specified. Default is None.
        cluster : compatible to :py:class:``~dask.disributed.LocalCluster`` or
        None
            This dask cluster is used to initialize a
            :py:class:``~dask.distributed.Client``, if no client is specified.
            Default is None.
        """
        self._check_client_cluster(client, cluster)
        if self._validate_client(client):
            self._client = client
            self._cluster = client.cluster
        else:
            self._cluster = cluster
            self._client = Client(cluster)

    def update_state(
            self,
            state: xr.DataArray,
            observations: Union[xr.Dataset, Iterable[xr.Dataset]],
            pseudo_state: xr.DataArray,
            analysis_time: pd.Timestamp
    ) -> xr.DataArray:
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
        pseudo_tensor = dask.delayed(torch.zeros(0).to(obs_state))

        logger.info('Normalise perturbations and observations')
        obs_cinv = self._get_chol_inverse(obs_cov)
        normed_perts, normed_obs = self._normalise_obs(pseudo_obs, obs_state,
                                                       obs_cinv)

        logger.info('Chunking and split background state')
        state = state.chunk(
            {'grid': self.chunksize, 'var_name': -1, 'time': -1, 'ensemble': -1}
        )
        state_grid = da.from_array(
            state['grid'].values, chunks=self.chunksize
        )
        chunk_pos = np.concatenate([[0], np.cumsum(state.chunks[-1])])
        state_mean, state_perts = state.state.split_mean_perts()

        logger.info('Scatter data')
        normed_perts, normed_obs, obs_grid = self.client.scatter(
            [normed_perts, normed_obs, obs_grid], broadcast=True
        )

        @dask.delayed
        def slice_data(array_to_slice, min_bound, max_bound):
            sliced_array = array_to_slice[..., min_bound:max_bound]
            return sliced_array

        @dask.delayed
        def to_tensor(array_to_convert, as_tensor):
            converted_tensor = torch.from_numpy(array_to_convert).to(as_tensor)
            return converted_tensor

        @dask.delayed
        def add_mean(perts, mean):
            added_mean = perts + mean.unsqueeze(dim=-2)
            return added_mean

        logger.info('Create analysis')
        analysis_list = []
        for k, pos in enumerate(chunk_pos[1:]):
            loc_perts = dask.delayed(slice_data)(
                state_perts.data, chunk_pos[k], pos
            )
            loc_perts = dask.delayed(to_tensor)(loc_perts, pseudo_tensor)
            loc_grid = dask.delayed(slice_data)(state_grid, chunk_pos[k], pos)
            loc_perts = dask.delayed(self.analyser)(
                loc_perts, normed_perts, normed_obs, loc_grid, obs_grid
            )
            loc_mean = dask.delayed(slice_data)(
                state_mean.data, chunk_pos[k], pos
            )
            loc_mean = dask.delayed(to_tensor)(loc_mean, pseudo_tensor)
            loc_ana = dask.delayed(add_mean)(loc_perts, loc_mean)
            analysis_list.append(loc_ana)

        @dask.delayed
        def cat_numpy(list_to_cat):
            concatenated_list = torch.cat(list_to_cat, dim=-1)
            concatenated_numpy = concatenated_list.numpy()
            return concatenated_numpy

        analysis = dask.delayed(cat_numpy)(analysis_list)
        analysis = analysis.compute()
        analysis = state.copy(data=analysis)
        return analysis


class DistributedLETKFCorr(CorrMixin, DistributedLETKFBase):
    """
    This is a dask-based implementation of the `localized ensemble transform
    Kalman filter` :cite:`hunt_efficient_2007` for correlated observations.
    This dask-based implementation is based on
    :py:class:``~dask.distributed.Client``.

    Parameters
    ----------
    client : :py:class:``~dask.distributed.Client`` or None
        This dask distributed client is used to parallelize the processes.
        Either this client or ``cluster`` has to be
        specified. Default is None. If both, cluster and client, are given, then
        client has priority.
    cluster : compatible to :py:class:``~dask.disributed.LocalCluster`` or None
        This dask cluster is used to initialize a :py:class:``~dask.distributed.
        Client``, if no client is specified.
        Either this cluster or a ``client`` has to be given. Default is None.
    chunksize : int, optional
        The data is splitted up such that every chunk has this number of
        samples. This influences the performance of this distributed version of
        the LETKF. Default is 10.
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
    smoother : bool, optional
        Indicates if this filter should be run in smoothing or in filtering
        mode. In smoothing mode, no analysis time is selected from given state
        and the ensemble weights are applied to the whole state. In filtering
        mode, the weights are applied only on selected analysis time. Default
        is False, indicating filtering mode.
    gpu : bool, optional
        Indicator if the weight estimation should be done on either GPU (True)
        or CPU (False): Default is None. For small models, estimation of the
        weights on CPU is faster than on GPU!.
    """
    def __str__(self):
        return 'Correlated {0:s}'.format(str(super(DistributedLETKFBase)))

    def __repr__(self):
        return 'Corr{0:s}'.format(repr(super(DistributedLETKFBase)))


class DistributedLETKFUncorr(UnCorrMixin, DistributedLETKFBase):
    """
    This is a dask-based implementation of the `localized ensemble transform
    Kalman filter` :cite:`hunt_efficient_2007` for uncorrelated observations.
    This dask-based implementation is based on
    :py:class:``~dask.distributed.Client``.

    Parameters
    ----------
    chunks : int, optional
        The data is splitted up in this number of chunks.
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
    smoother : bool, optional
        Indicates if this filter should be run in smoothing or in filtering
        mode. In smoothing mode, no analysis time is selected from given state
        and the ensemble weights are applied to the whole state. In filtering
        mode, the weights are applied only on selected analysis time. Default
        is False, indicating filtering mode.
    gpu : bool, optional
        Indicator if the weight estimation should be done on either GPU (True)
        or CPU (False): Default is None. For small models, estimation of the
        weights on CPU is faster than on GPU!.
    """
    def __str__(self):
        return 'Uncorrelated {0:s}'.format(str(super(DistributedLETKFBase)))

    def __repr__(self):
        return 'Uncorr{0:s}'.format(repr(super(DistributedLETKFBase)))
