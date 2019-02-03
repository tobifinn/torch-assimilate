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
import matplotlib
matplotlib.use('Agg')

# System modules
import logging
import numpy as np
import torch

# External modules
import matplotlib.pyplot as plt
import matplotlib.gridspec as mpl_gridspec

# Internal modules


logger = logging.getLogger(__name__)


def mse(a, b):
    return torch.mean(torch.sqrt((a.float() - b.float()) ** 2))


def get_metrics(autoencoder, prior_ens_0, prior_ens_1, obs, truth,):
    analysis, recon_obs = autoencoder.forward(
        observation=obs, prior=prior_ens_0
    )
    pseudo_obs_ens_1 = autoencoder.obs_operator(prior_ens_1)
    recon_prior_ens_1 = autoencoder.inference_net(
        observation=pseudo_obs_ens_1, prior=prior_ens_0,
    )
    prior_ens_1_mse = mse(recon_prior_ens_1, prior_ens_1)
    mean_absolute_increment = torch.mean(
        torch.abs(
            analysis.float() - prior_ens_0.float()
        )
    )
    truth_mse = mse(analysis, truth)
    obs_mse = mse(recon_obs, obs)

    metrics = {
        'gen/mean_absolute_increment': mean_absolute_increment.item(),
        'gen/loss_backward_mse': prior_ens_1_mse.item(),
        'gen/loss_analysis_mse': truth_mse.item(),
        'gen/obs_mse': obs_mse.item(),
    }
    return metrics


def write_metrics(summary_writer, metrics, global_step):
    for name, vals in metrics.items():
        summary_writer.add_scalar(
            name, np.mean(vals), global_step=global_step
        )


def plot_generator(model, dataset, device, _rnd, _run):
    idx_choice = _rnd.choice(len(dataset), size=1)
    sample = dataset[idx_choice]

    back_ens = sample['prior_ens'].view(-1, 40).float().to(device)
    ens_size = back_ens.size()[0]
    obs = sample['obs'].view(1, -1).float().to(device)
    obs_expanded = obs.expand(ens_size, -1)
    truth = sample['truth'].view(1, -1)

    forward_states = model.forward(observation=obs_expanded, prior=back_ens)
    analysed_ens = forward_states[0]

    plot_prior = back_ens.detach().cpu().numpy()
    plot_obs = obs.detach().cpu().numpy()
    plot_analysis = analysed_ens.detach().cpu().numpy()
    plot_truth = truth.detach().cpu().numpy()
    fig_cont, ax = plt.subplots(nrows=3, sharey=True, sharex=True)
    _ = ax[0].contourf(plot_prior, vmin=-2, vmax=2)
    _ = ax[1].contourf(plot_analysis, vmin=-2, vmax=2)
    _ = ax[2].contourf(plot_analysis-plot_prior, vmin=-2, vmax=2, cmap='coolwarm')
    ax[1].set_ylabel('Ensemble member')
    ax[1].set_xlabel('Grid point')

    prior_mean = np.mean(plot_prior, axis=0)
    analysis_mean = np.mean(plot_analysis, axis=0)
    truth_mean = np.mean(plot_truth, axis=0)
    gs = mpl_gridspec.GridSpec(4, 1)
    fig_lat_space = plt.Figure()
    ax1 = fig_lat_space.add_subplot(gs[:-1, :])
    ax1.plot(prior_mean, c='#1f77b4', label='prior', zorder=1)
    ax1.plot(analysis_mean, c='#ff7f0e', label='analysis', zorder=2)
    ax1.plot(truth_mean, c='0.5', alpha=0.5, label='truth', zorder=3)
    ax1.set_ylim(-3, 3)
    ax1.set_ylabel('Ensemble mean')
    ax1.legend(loc=3)
    ax2 = fig_lat_space.add_subplot(gs[-1, :], sharex=ax1)
    ax2.axhline(y=0, xmin=0, xmax=99, c='skyblue')
    try:
        for obs_loc in dataset.obs_operator._sel_obs_points:
            ax2.axvline(x=obs_loc, ymin=-1, ymax=1, c='0.5')
    except AttributeError:
        pass
    ax2.plot(analysis_mean - prior_mean, c='black')
    ax2.set_ylabel('Difference in mean')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks(ticks=(-1, 0, 1))
    ax2.set_xlabel('Grid point')
    ax2.set_xlim(0, 40)
    fig_lat_space.subplots_adjust(hspace=0.0)

    figures = {
        'ens_contour': fig_cont,
        'latent_space': fig_lat_space
    }

    return figures


def write_figures(model, data, device, summary_writer, global_step, _rnd, _run):
    figures = plot_generator(model, data, device=device,
                             _rnd=_rnd, _run=_run)
    for name, fig in figures.items():
        summary_writer.add_figure(
            name, fig, global_step=global_step, close=True
        )
