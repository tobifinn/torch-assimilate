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

# External modules
from distributed import Client, wait, Future
import dask
import dask.array as da

import torch

# Internal modules
from .letkf import LETKFCorr, localize_states
from .etkf_core import gen_weights_uncorr


logger = logging.getLogger(__name__)


@dask.delayed
def localize_state_chunkwise(state_grid, obs_grid, innov, hx_perts, obs_cov,
                             localization):
    if isinstance(state_grid, Future):
        state_grid = state_grid.result()
    localized_states = [
        localize_states(
            grid_l, obs_grid, innov, hx_perts, obs_cov, localization
        )
        for grid_l in state_grid
    ]
    return localized_states


@dask.delayed
def gen_weights_chunkwise(localized_states, back_prec, gen_weights_func):
    weights = []
    for state_l in localized_states:
        w_mean_l, w_perts_l = gen_weights_func(back_prec, *state_l)
        weights.append((w_mean_l + w_perts_l).t())
    weights = torch.stack(weights, dim=0).detach()
    return weights


@dask.delayed
def apply_weights_chunkwise(back_state, weights):
    ana_perts = torch.einsum(
        'ijkl,lkm->ijml', back_state, weights,
    ).detach()
    return ana_perts


class DistributedLETKFCorr(LETKFCorr):
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
    def __init__(self, client=None, cluster=None, chunksize=10,
                 localization=None, inf_factor=1.0, smoother=True,
                 gpu=False, pre_transform=None, post_transform=None):
        super().__init__(localization, inf_factor, smoother, gpu, pre_transform,
                         post_transform)
        self._cluster = None
        self._client = None
        self._chunksize = 1
        self.chunksize = chunksize
        self.set_client_cluster(client=client, cluster=cluster)

    @staticmethod
    def _validate_client(client):
        return isinstance(client, Client)

    @staticmethod
    def _validate_cluster(cluster):
        return hasattr(cluster, "scheduler_address")

    def _check_client_cluster(self, client, cluster):
        not_valid_cluster = not self._validate_cluster(cluster)
        not_valid_client = not self._validate_client(client)
        if not_valid_client and not_valid_cluster:
            raise ValueError(
                'Either a client or a cluster have to be specified!'
            )

    @property
    def cluster(self):
        return self._cluster

    @property
    def client(self):
        return self._client

    def set_client_cluster(self, client=None, cluster=None):
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

    def update_state(self, state, observations, pseudo_state, analysis_time):
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
        logger.info('####### DISTRIBUTED LETKF #######')
        logger.info('Starting with applying observation operator')
        innov, hx_perts, obs_cov, obs_grid = self._prepare(pseudo_state,
                                                           observations)
        logger.info('Chunking background state')
        state_mean, state_perts = state.state.split_mean_perts()
        state_perts = state_perts.chunk(
            {'grid': self.chunksize, 'var_name': -1, 'time': -1, 'ensemble': -1}
        )

        logger.info('Scatter the data to processes')
        ens_mems = len(state.ensemble)
        back_prec = self._get_back_prec(ens_mems)
        innov, hx_perts, obs_cov = self._states_to_torch(
            innov, hx_perts, obs_cov,
        )
        state_perts_data = state_perts.data
        state_grid = da.from_array(
            state_perts.grid.values, chunks=self.chunksize
        )
        persisted_computes = self.client.persist([state_perts_data, state_grid])
        state_perts_data, state_grid = self.client.gather(persisted_computes)
        wait([state_perts_data, state_grid])

        ana_perts = []
        for k, grid_block in enumerate(state_grid.blocks):
            localized_states = localize_state_chunkwise(
                grid_block, obs_grid, innov, hx_perts, obs_cov,
                self.localization
            )
            weights_l = gen_weights_chunkwise(localized_states, back_prec,
                                              self._gen_weights_func)
            torch_perts = dask.delayed(torch.as_tensor)(
                state_perts_data.blocks[..., k], dtype=weights_l.dtype
            )
            ana_perts_l = apply_weights_chunkwise(
                torch_perts, weights_l
            )
            ana_perts.append(ana_perts_l.detach().numpy())
        ana_perts = dask.delayed(da.concatenate)(ana_perts, axis=-1)

        logger.info('Create analysis perturbations')
        ana_perts = state_perts.copy(data=ana_perts.compute())

        logger.info('Create analysis')
        analysis = (ana_perts + state_mean).load()
        return analysis


class DistributedLETKFUncorr(DistributedLETKFCorr):
    """
    This is a MPI based implementation of the `localized ensemble transform
    Kalman filter` :cite:`hunt_efficient_2007` for uncorrelated observations.

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
    def __init__(self, client=None, cluster=None, chunksize=10,
                 localization=None, inf_factor=1.0,  smoother=True, gpu=False,
                 pre_transform=None, post_transform=None):
        super().__init__(client=client, cluster=cluster, chunksize=chunksize,
                         localization=localization, inf_factor=inf_factor,
                         smoother=smoother, gpu=gpu,
                         pre_transform=pre_transform,
                         post_transform=post_transform)
        self._gen_weights_func = gen_weights_uncorr
        self._correlated = False
