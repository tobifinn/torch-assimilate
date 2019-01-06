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

# External modules
import xarray as xr
import numpy as np
import torch
import torchvision

from sacred import Ingredient

# Internal modules
from pytassim.obs_ops.identity import IdentityOperator, obs_ingredient
from datasets import Lorenz96PreparedDataset, Lorenz96AssimDataset


logger = logging.getLogger(__name__)


data_ingredient = Ingredient('data', ingredients=[obs_ingredient, ])


@data_ingredient.config
def config():
    normalize = True
    base_data_path = '/scratch/local1/Data/neural_nets/neural_assim/data'
    rnd_pdf = 'normal'
    rnd_kwargs = dict()


class NormalizeSamples(object):
    def __init__(self, normalize_dict):
        self.normalize_dict = normalize_dict

    def _normalize(self, k, val):
        try:
            mean, std = self.normalize_dict[k]
            normalized_val = (val - mean) / std
        except KeyError:
            normalized_val = val
        return normalized_val

    def __call__(self, samples):
        normalized_samples = {
            k: self._normalize(k, val) for k, val in samples.items()
        }
        return normalized_samples


class SelectEnsMem(object):
    @data_ingredient.capture
    def __init__(self, _rnd, ens_size=50):
        self.rnd = _rnd
        self.ens_size = ens_size

    def __call__(self, samples):
        ens_mem = self.rnd.choice(self.ens_size, size=2, replace=False)
        prior_ens = samples['prior_ens'][..., ens_mem, :]
        return {
            'prior_ens': samples['prior_ens'],
            'prior_ens_0': prior_ens[..., 0, :],
            'prior_ens_1': prior_ens[..., 1, :],
            'obs': samples['obs'],
            'truth': samples['truth']
        }


def transform_to_tensor(samples):
    samples = {k: torch.from_numpy(v) for k, v in samples.items()}
    return samples


@data_ingredient.capture
def load_data(base_data_path, normalize, _run, _rnd, rnd_pdf, rnd_kwargs):
    train_ens_path = os.path.join(base_data_path, 'train_ens.nc')
    train_truth_path = os.path.join(base_data_path, 'train_vr1.nc')
    valid_ens_path = os.path.join(base_data_path, 'test_ens.nc')
    valid_truth_path = os.path.join(base_data_path, 'test_vr1.nc')

    obs_operator = IdentityOperator(obs_points=_run.config['obs']['obs_points'],
                                    random_state=_rnd)

    train_dataset = Lorenz96PreparedDataset(
        train_truth_path, train_ens_path, rnd=_rnd, obs_operator=obs_operator,
        rnd_pdf=rnd_pdf, rnd_kwargs=rnd_kwargs
    )

    transformers = []
    if normalize:
        normalize_dict = {
            'prior_ens': (
                train_dataset.ens_ds.mean(axis=(0, 1)),
                train_dataset.ens_ds.std(axis=(0, 1)),
            ),
            'obs': (
                train_dataset.obs_ds.mean(axis=0),
                train_dataset.obs_ds.std(axis=0),
            ),
            'truth': (
                train_dataset.ens_ds.mean(axis=(0, 1)),
                train_dataset.ens_ds.std(axis=(0, 1)),
            ),
        }
        _run.info['normalize_dict'] = normalize_dict
        transformers.append(
            NormalizeSamples(normalize_dict)
        )
    else:
        _run.info['normalize_dict'] = None
    transformers.append(transform_to_tensor)
    transformers.append(
        SelectEnsMem(ens_size=train_dataset.ens_size)
    )
    transformers = torchvision.transforms.Compose(transformers)
    train_dataset.transform = transformers

    test_dataset = Lorenz96PreparedDataset(
        valid_truth_path, valid_ens_path, rnd=_rnd, transform=transformers,
        obs_operator=obs_operator, rnd_pdf=rnd_pdf, rnd_kwargs=rnd_kwargs
    )

    assim_dataset = Lorenz96AssimDataset(
        _rnd, start_days=1000, end_days=100, dt_days=0.25,
        dt_obs=2, nr_grids=40, forcing=7.9, obs_operator=obs_operator,
        rnd_pdf=rnd_pdf, rnd_kwargs=rnd_kwargs
    )
    return train_dataset, test_dataset, assim_dataset
