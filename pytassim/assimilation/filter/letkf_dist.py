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
from concurrent.futures import as_completed
import itertools
from math import ceil

# External modules
import torch
import numpy as np
from tqdm import tqdm

# Internal modules
from .letkf import LETKFCorr, local_etkf


logger = logging.getLogger(__name__)


def local_etkf_batch(gen_weights_func, ind, innov, hx_perts, obs_cov, back_prec,
                     obs_grid, state_grid, state_perts, localization=None):
    ana_state = []
    weights = []
    w_mean = []
    for i in ind:
        ana_state_l, weights_l, w_mean_l = local_etkf(
            gen_weights_func, i, innov, hx_perts, obs_cov, back_prec, obs_grid,
            state_grid, state_perts, localization
        )
        ana_state.append(ana_state_l)
        weights.append(weights_l)
        w_mean.append(w_mean_l)
    ana_state = torch.stack(ana_state)
    weights = torch.stack(weights)
    w_mean = torch.stack(w_mean)
    return ana_state, weights, w_mean


class DistributedLETKFCorr(LETKFCorr):
    """
    This is a MPI based implementation of the `localized ensemble transform
    Kalman filter` :cite:`hunt_efficient_2007` for correlated observations.

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
    def __init__(self, pool, chunksize=10, localization=None, inf_factor=1.0,
                 smoother=True, gpu=False, pre_transform=None,
                 post_transform=None):
        super().__init__(localization, inf_factor, smoother, gpu, pre_transform,
                         post_transform)
        self._pool = None
        self.pool = pool
        self._chunksize = 1
        self.chunksize = chunksize

    @property
    def pool(self):
        return self._pool

    @pool.setter
    def pool(self, new_pool):
        if hasattr(new_pool, 'submit') and callable(new_pool.submit):
            self._pool = new_pool
        else:
            raise TypeError('Given distributed pool needs a '
                            'concurrent.futures-like `submit` method!')

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
        logger.info('####### DISTRIBUTED LETKF #######')
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
        grid_inds = list(self._slice_data(range(len_state_grid)))
        processes = []
        total_steps = ceil(len_state_grid/self.chunksize)
        logger.info('Starting with job submission')
        for ind in tqdm(grid_inds, total=total_steps):
            tmp_process = self.pool.submit(
                local_etkf_batch, self._gen_weights_func, ind, innov, hx_perts,
                obs_cov, back_prec, obs_grid, state_grid, back_state,
                self.localization
            )
            processes.append(tmp_process)

        logger.info('Waiting until jobs are finished')
        for _ in tqdm(as_completed(processes), total=total_steps):
            pass

        logger.info('Gathering the analysis')
        state_perts.values = torch.cat(
            [p.result()[0] for p in tqdm(processes, total=total_steps)], dim=0
        ).numpy()
        analysis = (state_mean+state_perts).transpose(*state.dims)
        logger.info('Finished with analysis creation')
        return analysis

    def _slice_data(self, data):
        data = iter(data)
        while True:
            chunk = tuple(itertools.islice(data, self.chunksize))
            if not chunk:
                return
            else:
                yield chunk

    @staticmethod
    def _share_states(*states):
        shared_states = [s.share_memory_() for s in states]
        return shared_states
