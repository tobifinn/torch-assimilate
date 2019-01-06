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
import logging
import os
import time
import json

# External modules
import torch

import numpy as np

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.stflow import LogFileWriter
from sacred.utils import apply_backspaces_and_linefeeds

import pymongo

import tensorboardX

from tqdm import tqdm

# Internal modules
from data_loader import data_ingredient, load_data
from models_cycle import model_ingredient, Model
from eval_utils import write_figures, get_metrics, write_metrics
from test_model import test_model


logger = logging.getLogger(__name__)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_mongo_observer(config_file):
    with open(config_file, mode='r') as fh:
        json_conf = json.load(fh)
    observer = MongoObserver.create(
        url="{0:s}:{1:d}".format(json_conf['server'], json_conf['port']),
        db_name=json_conf['database'],
        username=json_conf['username'],
        password=json_conf['password'],
        authSource=json_conf['database'],
        authMechanism='SCRAM-SHA-1',
        serverSelectionTimeoutMS=1000
    )
    _ = observer.metrics.database.client.server_info()
    return observer


exp = Experiment(
    'test_cycle',
    ingredients=[data_ingredient, model_ingredient]
)

# try:
#    exp.observers.append(get_mongo_observer('mongodb.json'))
# except pymongo.errors.ServerSelectionTimeoutError as e:
#    logger.warning('######### WARNING #########\n'
#                   'Run will not be saved in a database, because MongoDB '
#                   'server is not available!')
# exp.captured_out_filter = apply_backspaces_and_linefeeds


@exp.config
def config():
    log_path = '/scratch/local1/Data/neural_nets/neural_assim'
    load_path = None
    batch_size = 64
    epochs = 100
    lam_gan_decay


def out_as_str(model_out):
    loss_list = ['{0:s}:{1:.04f}'.format(key, val)
                 for key, val in model_out['losses'].items()]
    metric_list = ['{0:s}:{1:.04f}'.format(key, val)
                   for key, val in model_out['metrics'].items()]
    out_list = loss_list + metric_list
    return ','.join(out_list)


@exp.capture
def log_metric(model_output, step, _run, valid=False):
    if valid:
        prefix = 'valid'
    else:
        prefix = 'train'
    for c, c_dict in model_output.items():
        for m, m_val in c_dict.items():
            full_name = '{0:s}.{1:s}.{2:s}'.format(prefix, c, m)
            _run.log_scalar(full_name, m_val, step)


@exp.capture
def train_model(model, train_data, valid_data, assim_ds, summary_writers,
                log_path, batch_size, epochs, _log, _run, _rnd):
    _log.info('Starting to train the model')
    model_path = os.path.join(log_path, 'models')
    save_path = os.path.join(model_path, '{0:s}_{1:s}'.format(
        _run.experiment_info['name'], _run.start_time.strftime('%Y%m%d%H%M')
    ))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_generator = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    valid_generator = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=True, num_workers=1
    )

    optimizers = model.get_optimizers()

    write_figures(model, valid_data, device, summary_writers['test'],
                  0, _rnd, _run)

    test_model(model, assim_ds, summary_writers['test'], 0, _rnd, _run)

    iters_p_epoch = len(train_data)//batch_size - 1
    lam_gan_decay = 20
    del_lam = 1 / lam_gan_decay
    n_iters = 0
    best_loss = 99999999999
    tot_pbar = tqdm(total=epochs, desc='Total')
    for epoch in range(epochs):
        e_pbar = tqdm(total=iters_p_epoch, desc='Epoch',
                      leave=False)
        model.train()
        curr_lam = min(1., del_lam * epoch)

        summary_writers['train'].add_scalar(
            'observer/lam_gan', curr_lam, n_iters
        )

        for nr_sample, train_sample in enumerate(train_generator):
            prior_ens_0 = train_sample['prior_ens_0'].float().to(device)
            prior_ens_1 = train_sample['prior_ens_1'].float().to(device)
            obs = train_sample['obs'].float().to(device)
            #truth = train_sample['truth'].to(device)

            optimizers['disc_prior'].zero_grad()
            train_loss_disc_prior, _ = model.loss_disc_prior(
                prior_ens_0, prior_ens_1, obs
            )
            train_loss_disc_prior['tot_loss'].backward()
            optimizers['disc_prior'].step()

            optimizers['disc_obs'].zero_grad()
            train_loss_disc_obs, _ = model.loss_disc_obs(
                prior_ens_0, prior_ens_1, obs
            )
            train_loss_disc_obs['tot_loss'].backward()
            optimizers['disc_obs'].step()

            summary_writers['train'].add_scalar(
                'loss/disc_prior', train_loss_disc_prior['tot_loss'].item(),
                global_step=n_iters+1
            )
            summary_writers['train'].add_scalar(
                'loss/disc_obs', train_loss_disc_obs['tot_loss'].item(),
                global_step=n_iters+1
            )

            if n_iters % model.disc_steps == 0:
                optimizers['gen'].zero_grad()
                train_loss_gen_for, _ = model.loss_gen_forward(
                    prior_ens_0, prior_ens_1, obs, lam_gan=curr_lam)
                train_loss_gen_for['tot_loss'].backward()
                train_loss_gen_back, _ = model.loss_gen_backward(
                    prior_ens_0, prior_ens_1, obs, lam_gan=curr_lam)
                train_loss_gen_back['tot_loss'].backward()
                optimizers['gen'].step()

                summary_writers['train'].add_scalar(
                    'loss/gen',
                    train_loss_gen_for['tot_loss'].item() +
                    train_loss_gen_back['tot_loss'].item(),
                    global_step=n_iters+1
                )
                summary_writers['train'].add_scalar(
                    'gen/loss_for',
                    train_loss_gen_for['tot_loss'].item(),
                    global_step=n_iters+1
                )
                summary_writers['train'].add_scalar(
                    'gen/loss_back',
                    train_loss_gen_back['tot_loss'].item(),
                    global_step=n_iters+1
                )

            e_pbar.set_postfix(
                loss_gen=train_loss_gen_for['tot_loss'].item() +
                         train_loss_gen_back['tot_loss'].item(),
                loss_disc_prior=train_loss_disc_prior['tot_loss'].item(),
                loss_disc_obs=train_loss_disc_obs['tot_loss'].item()
            )

            if n_iters % 500 == 0:
                metric_dict = {}
                metric_dict.update(
                    {'disc_prior/{0:s}'.format(n): loss.item()
                     for n, loss in train_loss_disc_prior.items()
                     if n[:4] == 'loss'}
                )
                metric_dict.update(
                    {'disc_obs/{0:s}'.format(n): loss.item()
                     for n, loss in train_loss_disc_obs.items()
                     if n[:4] == 'loss'}
                )
                write_metrics(summary_writers['train'], metric_dict, n_iters,
                              'disc')

                metric_dict = {}
                metric_dict.update(
                    {'gen_forward/{0:s}'.format(n): loss.item()
                     for n, loss in train_loss_gen_for.items()
                     if n[:4] == 'loss'}
                )
                metric_dict.update(
                    {'gen_backward/{0:s}'.format(n): loss.item()
                     for n, loss in train_loss_gen_back.items()
                     if n[:4] == 'loss'}
                )
                write_metrics(summary_writers['train'], metric_dict, n_iters,
                              'gen')
            n_iters += 1
            e_pbar.update()

        model.eval()
        with torch.set_grad_enabled(False):
            test_loss = {'disc_obs': [], 'disc_prior': [], 'gen': [],}
            for valid_sample in valid_generator:
                prior_ens_0 = valid_sample['prior_ens_0'].float().to(device)
                prior_ens_1 = valid_sample['prior_ens_1'].float().to(device)
                obs = valid_sample['obs'].float().to(device)
                #truth = valid_sample['truth'].to(device)

                test_loss_for, _ = model.loss_gen_forward(
                    prior_ens_0, prior_ens_1, obs,)
                test_loss_back, _ = model.loss_gen_backward(
                    prior_ens_0, prior_ens_1, obs,)
                test_loss_gen = test_loss_for['tot_loss']+test_loss_back['tot_loss']
                test_loss['gen'].append(test_loss_gen.item())

                test_loss_disc_prior, _ = model.loss_disc_prior(
                    prior_ens_0, prior_ens_1, obs,)
                test_loss['disc_prior'].append(
                    test_loss_disc_prior['tot_loss'].item())

                test_loss_disc_obs, _ = model.loss_disc_obs(
                    prior_ens_0, prior_ens_1, obs, )
                test_loss['disc_obs'].append(
                    test_loss_disc_obs['tot_loss'].item())
            test_loss = {k: np.mean(l) for k, l in test_loss.items()}
            for name, loss in test_loss.items():
                summary_writers['test'].add_scalar(
                    'loss/{0:s}'.format(name), test_loss[name],
                    global_step=n_iters
                )
        write_figures(model, valid_data, device, summary_writers['test'],
                      n_iters, _rnd, _run)
        test_model(model, assim_ds, summary_writers['test'],
                   n_iters, _rnd, _run)
        if test_loss['gen'] < best_loss:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizers': {k: opt.state_dict()
                                   for k, opt in optimizers.items()}
                }, os.path.join(save_path, 'model_best.tar')
            )
            best_loss = test_loss['gen']

        e_pbar.close()

        tot_pbar.set_postfix(loss_gen=test_loss['gen'],
                             loss_disc_prior=test_loss['disc_prior'],
                             loss_disc_obs=test_loss['disc_obs'])
        tot_pbar.update()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@exp.capture
@exp.automain
def run_experiment(log_path, _run, _log, _rnd):
    summary_dir = os.path.join(
        log_path, 'tensorboard', '{0:s}_{1:s}'.format(
            _run.experiment_info['name'],
            _run.start_time.strftime('%Y%m%d%H%M')
        )
    )
    train_dataset, valid_dataset, assim_dataset = load_data()
    _log.info('Loaded the data')

    model = Model(obs_size=len(train_dataset.obs_grid)).to(device)
    _log.info('Num model parameters:{0:d}'.format(count_parameters(model)))
    _log.info('Compiled the model')

    train_writer = tensorboardX.SummaryWriter(summary_dir + '/train')
    test_writer = tensorboardX.SummaryWriter(summary_dir + '/test')
    train_model(model, train_data=train_dataset, valid_data=valid_dataset,
                assim_ds=assim_dataset,
                summary_writers={'train': train_writer, 'test': test_writer})

    _log.info('Finished training of {0:s}_{1:s}'.format(
       _run.experiment_info['name'], _run.start_time.strftime('%Y%m%d%H%M')
    ))
