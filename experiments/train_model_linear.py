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
from models_linear import model_ingredient, Model
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
    'test',
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
        valid_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    write_figures(model, valid_data, device, summary_writers['test'],
                  0, _rnd, _run)

    test_model(model, assim_ds, summary_writers['test'], 0, _rnd, _run)

    iters_p_epoch = len(train_data)//batch_size - 1
    n_iters = 0
    best_loss = 99999999999
    tot_pbar = tqdm(total=epochs, desc='Total')
    for epoch in range(epochs):
        e_pbar = tqdm(total=iters_p_epoch, desc='Epoch', leave=False)
        for nr_sample, train_sample in enumerate(train_generator):
            prior_ens_0 = train_sample['prior_ens_0'].float().to(device)
            prior_ens_1 = train_sample['prior_ens_1'].float().to(device)
            obs = train_sample['obs'].float().to(device)
            losses_disc = model.trainstep_disc(
                prior_ens_0, prior_ens_1, obs
            )
            if n_iters % model.disc_steps == 0:
                losses_gen = model.trainstep_gen(
                        prior_ens_0, prior_ens_1, obs,)

            e_pbar.set_postfix(loss_gen=losses_gen['tot_loss'].item(),
                               loss_disc=losses_disc['tot_loss'].item())

            summary_writers['train'].add_scalar(
                'loss/disc', losses_disc['tot_loss'].item(),
                global_step=n_iters+1
            )
            summary_writers['train'].add_scalar(
                'loss/gen', losses_gen['tot_loss'].item(),
                global_step=n_iters+1
            )
            if n_iters % 500 == 0:
                summary_writers['train'].add_scalar(
                    'gen/sigma_0_l1',
                    model.enc_net.data_net[0][0].weights_sigma.item(),
                    global_step=n_iters+1
                )
                summary_writers['train'].add_scalar(
                    'disc/sigma_0_l1',
                    model.disc_prior.net[0][0].weights_sigma.item(),
                    global_step=n_iters+1
                )
                write_metrics(summary_writers['train'], losses_disc, n_iters,
                              'disc')
                write_metrics(summary_writers['train'], losses_gen, n_iters,
                              'gen')
            e_pbar.update()
            n_iters += 1

        with torch.set_grad_enabled(False):
            test_loss = {'disc': [], 'gen': []}
            for valid_sample in valid_generator:
                prior_ens_0 = valid_sample['prior_ens_0'].float().to(device)
                prior_ens_1 = valid_sample['prior_ens_1'].float().to(device)
                obs = valid_sample['obs'].float().to(device)

                test_loss_disc = model.eval_disc(prior_ens_0, prior_ens_1, obs)
                test_loss_gen = model.eval_gen(prior_ens_0, prior_ens_1, obs)

                test_loss['gen'].append(test_loss_gen['tot_loss'].item())
                test_loss['disc'].append(test_loss_disc['tot_loss'].item())

            test_loss = {k: np.mean(l) for k, l in test_loss.items()}
            summary_writers['test'].add_scalar(
                'loss/disc', test_loss['disc'],
                global_step=n_iters
            )
            summary_writers['test'].add_scalar(
                'loss/gen', test_loss['gen'],
                global_step=n_iters
            )

        write_figures(model, valid_data, device, summary_writers['test'],
                      n_iters, _rnd, _run)
        test_model(model, assim_ds, summary_writers['test'],
                   n_iters, _rnd, _run)
        if test_loss['gen'] < best_loss:
            model_path = os.path.join(
                save_path, 'model_latest_{0:d}.pt'.format(epoch)
            )
            torch.save(model, model_path)
            best_loss = test_loss['gen']

        tot_pbar.set_postfix(loss_gen=test_loss['gen'],
                             loss_disc=test_loss['disc'])
        tot_pbar.update()
        e_pbar.close()


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

    model = Model().to(device)
    for k, i in enumerate(train_dataset.obs_operator._sel_obs_points):
        model.dec_net.weight.data[k, i] = 1.
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

