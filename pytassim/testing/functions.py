#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 04.01.21
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging

# External modules
import numpy as np
import xarray as xr


# Internal modules


logger = logging.getLogger(__name__)


def generate_random_weights(ens_size: int = 10) -> xr.DataArray:
    weights_mean = np.random.normal(size=(ens_size, 1))
    weights_perts = np.random.normal(size=(ens_size, ens_size))
    weights_perts -= weights_perts.mean(axis=1, keepdims=True)
    weights = weights_mean + weights_perts + np.eye(ens_size)
    xr_weights = xr.DataArray(
        weights,
        coords={
            'ensemble': np.arange(10),
            'ensemble_new': np.arange(10)
        },
        dims=['ensemble', 'ensemble_new']
    )
    return xr_weights
