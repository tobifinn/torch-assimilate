#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12/11/18
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
import warnings

import matplotlib
matplotlib.use('Agg')

import logging

# External modules
import numpy as np
import xarray as xr
import torch
import torch.utils.data
from tqdm import tqdm

import matplotlib.pyplot as plt

# Internal modules
from pytassim.assimilation.neural import NeuralAssimilation
from pytassim.transform.normalize import Normalizer
from pytassim.assimilation.filter import LETKFilter
from pytassim.localization import GaspariCohn
from pytassim.model.forward_model import forward_model
from pytassim.model import Lorenz96
from pytassim.model.integration import RK4Integrator


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def l96_distance(a, b):
    abs_dist = np.abs(a-b)
    dist = np.stack([abs_dist, 40-abs_dist]).min(axis=0)
    return dist


def create_ensemble(dataset, _rnd):
    ens_size = 50

    ens_f = torch.tensor(
        _rnd.normal(0, 0.5, size=(1, ens_size, 1)) + dataset.forcing
    )

    # Intialize the ensemble model and the integrator
    l96_ensemble = Lorenz96(ens_f)
    ensemble_integrator = RK4Integrator(l96_ensemble, dt=dataset.dt)

    # Perturbations of the ensemble initial state are roundabout 10 % of the
    # interspatial variability of VR1
    ens_pert_std = 2.0

    # Our forecast time is five days to get the same amount of samples as in VR1
    ens_lead_time = 5

    ens_fcst_steps = np.arange(0, ens_lead_time, dataset.dt_days)

    start_state = dataset.ds_vr1.isel(time=0)

    ens_start_pert = _rnd.normal(0, ens_pert_std,
                                 size=(1, ens_size, dataset.nr_grids))
    ens_start_state = xr.DataArray(
        data=ens_start_pert,
        coords=dict(
            var_name=['x', ],
            grid=np.arange(dataset.nr_grids),
            ensemble=np.arange(ens_size)
        ),
        dims=['var_name', 'ensemble', 'grid', ]
    )
    ens_start_state += start_state.values
    return ensemble_integrator, ens_start_state, ens_fcst_steps


def create_assimilation(assimilation, ds_forecasts, start_state, integrator,
                        dt_ana, obs_dataset, desc):
    ens_fcst_steps = ds_forecasts.time.values
    first_guess = None
    latest_state = start_state
    p_bar = tqdm(total=len(obs_dataset), desc=desc, leave=False)
    for obs in obs_dataset:
        if first_guess is not None:
            analysis = assimilation.assimilate(
                first_guess, obs, analysis_time=obs.time.values
            )
            latest_state = analysis.squeeze(dim='var_name')
        ensemble_forecast = forward_model(
            ens_fcst_steps, 0,
            torch.tensor(latest_state.values),
            integrator,
        )
        ds_forecasts.loc[dict(analysis=obs.time.values)] = ensemble_forecast
        first_guess = ensemble_forecast.sel(
            time=[dt_ana, ], drop=False
        )
        first_guess['time'] += obs.time.values
        p_bar.update()
    p_bar.close()
    return ds_forecasts


def get_error_ens_mean(forecasts, truth):
    stacked_fcsts = forecasts.stack(
        stacked=['analysis', 'time']
    )
    stacked_fcsts['validtime'] = stacked_fcsts['analysis'] + \
                                 stacked_fcsts['time']
    intersect_bool = [True if t in truth.time.values else False for t in
                      stacked_fcsts['validtime'].values]
    stacked_fcsts = stacked_fcsts.sel(stacked=intersect_bool)
    sliced_truth = truth.sel(time=stacked_fcsts.validtime.values)
    sliced_truth = sliced_truth.rename(time='stacked')
    sliced_truth['stacked'] = stacked_fcsts['stacked']
    sliced_truth = sliced_truth.unstack('stacked')
    error = forecasts.mean('ensemble') - sliced_truth.mean('ensemble')
    return error


def plot_member(fcst_mem, ax, *args, **kwargs):
    for analysis in fcst_mem.analysis.values:
        validtime = fcst_mem['time'] + analysis
        mem_plot = ax.plot(validtime,
                           fcst_mem.sel(analysis=analysis).squeeze(), *args,
                           **kwargs)
    return mem_plot


def test_model(model, dataset, summary_writer, global_step, _rnd, _run):
    # LETKF
    gaspari_cohn = GaspariCohn(length_scale=5, dist_func=l96_distance)
    letkf = LETKFilter(localization=gaspari_cohn, inf_factor=1.1, gpu=False)

    # Neural assimilation
    try:
        normalizer = Normalizer(
            ens_stat=_run.info['normalize_dict']['prior_ens'],
            obs_stat=(_run.info['normalize_dict']['obs'],)
        )
        pre_transform = (normalizer, )
        post_transform = (normalizer, )
    except KeyError:
        pre_transform = None
        post_transform = None

    neural_assimilation = NeuralAssimilation(
        model=model, gpu=True, pre_transform=pre_transform,
        post_transform=post_transform
    )

    ens_int, ens_start_state, ens_fcst_steps = create_ensemble(dataset, _rnd)

    zero_data = np.zeros(
        (1, len(dataset.ds_obs.time.values), len(ens_fcst_steps),
         len(ens_start_state.ensemble), dataset.nr_grids))
    letkf_forecasts = xr.DataArray(
        data=zero_data,
        coords={
            'var_name': ens_start_state.var_name.values,
            'analysis': dataset.ds_obs.time.values,
            'time': ens_fcst_steps,
            'ensemble': ens_start_state.ensemble.values,
            'grid': ens_start_state.grid.values,
        },
        dims=['var_name', 'analysis', 'time', 'ensemble', 'grid']
    )
    neural_forecasts = xr.DataArray(
        data=zero_data.copy(),
        coords={
            'var_name': ens_start_state.var_name.values,
            'analysis': dataset.ds_obs.time.values,
            'time': ens_fcst_steps,
            'ensemble': ens_start_state.ensemble.values,
            'grid': ens_start_state.grid.values,
        },
        dims=['var_name', 'analysis', 'time', 'ensemble', 'grid']
    )

    data_loader = dataset

    # Create forecasts
    neural_forecasts = create_assimilation(
        neural_assimilation, neural_forecasts, ens_start_state,
        ens_int, dataset.dt_obs, data_loader, 'Neural'
    )
    letkf_forecasts = create_assimilation(
        letkf, letkf_forecasts, ens_start_state, ens_int,
        dataset.dt_obs, data_loader, 'LETKF'
    )

    letkf_err = get_error_ens_mean(letkf_forecasts, dataset.ds_vr1)
    neural_err = get_error_ens_mean(neural_forecasts, dataset.ds_vr1)

    letkf_rmse = np.sqrt((letkf_err ** 2).mean(['analysis', 'grid']))
    letkf_mae = np.abs(letkf_err).mean(['analysis', 'grid'])
    letkf_std = letkf_forecasts.std('ensemble').mean(['analysis', 'grid'])

    neural_rmse = np.sqrt((neural_err ** 2).mean(['analysis', 'grid']))
    neural_mae = np.abs(neural_err).mean(['analysis', 'grid'])
    neural_std = neural_forecasts.std('ensemble').mean(['analysis', 'grid'])

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots()
    ax.plot(letkf_std['time'], letkf_std.squeeze(), label='LETKF Spread',
            c=color_cycle[0], ls='--')
    ax.plot(letkf_mae['time'], letkf_mae.squeeze(), label='LETKF MAE',
            c=color_cycle[0], ls='-.')
    ax.plot(letkf_rmse['time'], letkf_rmse.squeeze(), label='LETKF RMSE',
            c=color_cycle[0])
    ax.plot(neural_std['time'], neural_std.squeeze(), label='Neural Spread',
            c=color_cycle[1], ls='--')
    ax.plot(neural_mae['time'], neural_mae.squeeze(), label='Neural MAE',
            c=color_cycle[1], ls='-.')
    ax.plot(neural_rmse['time'], neural_rmse.squeeze(), label='Neural RMSE',
            c=color_cycle[1])
    ax.set_ylim(0, 5)
    ax.set_xlabel('Lead time in days')
    ax.set_ylabel('Error')
    ax.legend()

    summary_writer.add_figure(
        'fcst_error', fig, global_step=global_step, close=True
    )

    start_time = float(_rnd.choice(dataset.ds_obs.time.values[:-10], size=1))
    end_time = start_time + 20

    plot_grid_point = _rnd.choice(ens_start_state.grid.values, size=1)
    plot_letkf = letkf_forecasts.isel(time=slice(0, 9)).sel(
        analysis=slice(start_time, end_time), grid=plot_grid_point)
    plot_neural = neural_forecasts.isel(time=slice(0, 9)).sel(
        analysis=slice(start_time, end_time), grid=plot_grid_point)
    plot_vr1 = dataset.ds_vr1.sel(time=slice(start_time, end_time + 5),
                                  grid=plot_grid_point)

    fig, ax = plt.subplots()
    _ = plot_member(plot_letkf, ax, c=color_cycle[0], alpha=0.1)
    _ = plot_member(plot_neural, ax, c=color_cycle[1], alpha=0.1)
    plt_letkf, = plot_member(plot_letkf.mean('ensemble'), ax, label='LETKF',
                             c=color_cycle[0])
    plt_neural, = plot_member(plot_neural.mean('ensemble'), ax, label='Neural',
                              c=color_cycle[1])
    plt_truth, = ax.plot(plot_vr1['time'], plot_vr1.squeeze(), c='black')
    ax.set_ylim(-12, 12)
    ax.set_xlim(start_time, end_time)
    ax.set_ylabel('Value (Grid point: {0:d})'.format(int(plot_grid_point)))
    ax.set_xlabel('time')
    ax.legend(
        [plt_truth, plt_letkf, plt_neural], ['Truth', 'LETKF', 'Neural net']
    )

    summary_writer.add_figure(
        'time_series', fig, global_step=global_step, close=True
    )
