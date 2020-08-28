#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12/10/18
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
from typing import Type

# External modules
from tqdm import tqdm
import xarray as xr
import numpy as np
import torch

# Internal modules
from pytassim.model.integration.integrator import BaseIntegrator


logger = logging.getLogger(__name__)


def forward_model(
        all_steps: np.ndarray,
        start_point: int,
        start_state: torch.Tensor,
        integrator: Type[BaseIntegrator]
) -> xr.DataArray:
    """
    This forward model can be used to integrate the Lorenz 96 in time.

    Parameters
    ----------
    all_steps : :py:class:`np.ndarray`
        All time steps, which should be estimated.
    start_point : int
        Starting from this point, all steps are written as output.
    start_state : :py:class:`torch.Tensor`
        This start state is used as initial value for the integration.
    integrator : child of :py:class:`pytassim.model.integration.integrator.BaseIntegrator
        This integrator is used to integrate the model.
    """
    output_steps = all_steps[all_steps >= start_point]
    dataset = []
    curr_state = start_state
    for t in tqdm(all_steps, leave=False):
        curr_state = integrator.integrate(curr_state)
        if t in output_steps:
            dataset.append(curr_state)
    stacked_dataset = torch.stack(dataset, dim=-1).numpy()
    stacked_dataset = np.array(stacked_dataset, ndmin=4)
    stacked_dataset = stacked_dataset.transpose(0, 3, 1, 2)
    _, _, ens_size, nr_grid = stacked_dataset.shape
    dataset = xr.DataArray(
        data=stacked_dataset,
        coords={
            'var_name': ['x', ],
            'time': output_steps,
            'ensemble': np.arange(ens_size),
            'grid': np.arange(nr_grid)
        },
        dims=['var_name', 'time', 'ensemble', 'grid']
    )
    return dataset
