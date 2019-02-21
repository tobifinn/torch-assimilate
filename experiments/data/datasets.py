#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12/12/18
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
import torch
import torch.utils.data
import xarray as xr
import numpy as np

# Internal modules
from pytassim.model import Lorenz96
from pytassim.model.integration import RK4Integrator
from pytassim.model.lorenz_96.forward_model import forward_model

logger = logging.getLogger(__name__)


class Lorenz96AssimDataset(torch.utils.data.Dataset):
    def __init__(self, rnd, start_days=1000, end_days=730, dt_days=0.25,
                 dt_obs=2, nr_grids=40, forcing=7.9, obs_operator=None,
                 rnd_pdf='normal', obs_var=0.5, rnd_kwargs=None):
        self.rnd = rnd
        self.start_days = start_days
        self.end_days = end_days
        self.dt_days = dt_days
        self.dt_obs = dt_obs
        self.nr_grids = nr_grids
        self.forcing = forcing
        self.obs_operator = obs_operator
        self.rnd_pdf = rnd_pdf
        self.obs_var = obs_var
        if isinstance(rnd_kwargs, dict):
            self.rnd_kwargs = rnd_kwargs
        else:
            self.rnd_kwargs = {}
        self.ds_vr1 = self._create_vr1()
        self.ds_obs = self._create_obs(self.ds_vr1)

    def __len__(self):
        return len(self.ds_obs.time[:-1])

    def __getitem__(self, item):
        sel_obs = self.ds_obs.isel(time=[item, ])
        sel_obs.obs.operator = self.obs_operator.get_obs_method
        return sel_obs

    @property
    def dt(self):
        return self.dt_days / 5

    def _create_vr1(self):
        all_steps = np.arange(0, self.end_days+self.dt_days+self.start_days,
                              self.dt_days)
        start_state = self.rnd.normal(0, 0.01, size=(1, self.nr_grids))
        l96_vr1 = Lorenz96(forcing=self.forcing)
        vr1_integrator = RK4Integrator(l96_vr1, dt=self.dt)
        ds_vr1 = forward_model(
            all_steps, self.start_days, torch.tensor(start_state),
            vr1_integrator
        )
        return ds_vr1

    def _create_obs(self, truth):
        obs_times = np.arange(
            self.start_days, self.start_days + self.end_days + self.dt_obs,
            self.dt_obs
        )
        truth_sliced = truth.sel(time=obs_times)
        rnd_noise = getattr(self.rnd, self.rnd_pdf)(
            size=truth_sliced.shape, **self.rnd_kwargs
        )
        truth_sliced = truth_sliced + rnd_noise

        obs_state = self.obs_operator(truth_sliced).squeeze()
        obs_state = obs_state.rename(grid='obs_grid_1')
        obs_cov = xr.DataArray(
            data=[self.obs_var, ] * len(obs_state.obs_grid_1),
            coords={
                'obs_grid_1': obs_state.obs_grid_1.values,
            },
            dims=('obs_grid_1', )
        )
        ds_obs = xr.Dataset(
            {
                'observations': obs_state,
                'covariance': obs_cov
            }
        )
        return ds_obs


class Lorenz96PreparedDataset(torch.utils.data.Dataset):
    def __init__(self, truth_file, ens_file, obs_operator, rnd,
                 transform=None, rnd_pdf='normal', rnd_kwargs=None):
        self.rnd = rnd
        self.rnd_pdf = rnd_pdf
        if isinstance(rnd_kwargs, dict):
            self.rnd_kwargs = rnd_kwargs
        else:
            self.rnd_kwargs = dict()
        self.obs_operator = obs_operator
        self.transform = transform

        self.ens_ds = self._load_ens(ens_file)
        self.ens_size = len(self.ens_ds.ensemble)
        self.times = self.ens_ds['validtime'].values

        raw_truth = xr.open_dataarray(truth_file).squeeze()
        self.truth_ds = raw_truth.sel(time=self.ens_ds['validtime'].values)
        self.truth_ds = self.truth_ds.rename(time='samples')
        self.obs_ds = self._create_obs()
        self.obs_grid = self.obs_ds.obs_grid_1.values

        self.ens_ds = self.ens_ds.values
        self.truth_ds = self.truth_ds.values
        self.obs_ds = self.obs_ds.values

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        if self.obs_ds is None:
            raise ValueError('Please call first `create_obs`!')
        drawn_ens = self.ens_ds[index]
        drawn_obs = self.obs_ds[index]
        drawn_truth = self.truth_ds[index]
        sample = {
            'prior_ens': drawn_ens,
            'obs': drawn_obs,
            'truth': drawn_truth
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def _load_ens(path):
        raw_ens = xr.open_dataarray(path).isel(analysis=slice(None, -1))
        raw_ens = raw_ens.squeeze()
        stacked_ens = raw_ens.stack(samples=['analysis', 'time'])
        stacked_ens['validtime'] = stacked_ens['analysis'] + stacked_ens['time']
        trans_ens = stacked_ens.transpose('samples', 'ensemble', 'grid', )
        return trans_ens

    def _prepare_obs(self, obs, rnd_pdf, rnd_kwargs):
        rnd_noise = getattr(self.rnd, rnd_pdf)(
            size=obs.shape, **rnd_kwargs
        )
        obs_state = obs + rnd_noise
        obs_state = obs_state.rename(grid='obs_grid_1')
        return obs_state

    def _create_obs(self):
        rnd_noise = getattr(self.rnd, self.rnd_pdf)(
            size=self.truth_ds.shape, **self.rnd_kwargs
        )
        tmp_ds = self.truth_ds + rnd_noise
        raw_obs = self.obs_operator(tmp_ds)
        obs_state = raw_obs.rename(grid='obs_grid_1')
        return obs_state
