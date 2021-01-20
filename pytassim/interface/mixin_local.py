#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 18.12.20
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Tuple

# External modules
import xarray as xr
import pandas as pd
import numpy as np
import torch

# Internal modules
import pytassim.utilities.pandas as utils


logger = logging.getLogger(__name__)


class DomainLocalizedMixin(object):
    @property
    def localized_module(self):
        def wrapper(grid_info, *args, obs_info=None, args_to_skip=None):
            if self.localization is not None:
                luse, lweights = self.localization.localize_obs(
                    grid_info, obs_info
                )
                lweights = np.sqrt(lweights[luse])
                if args_to_skip is None:
                    args_to_skip = []
                args = [
                    arg if k in args_to_skip else arg[..., luse] * lweights
                    for k, arg in enumerate(args)
                ]
            return self.module(*args)
        return wrapper

    @staticmethod
    def _extract_obs_information(observations: xr.DataArray) -> pd.DataFrame:
        obs_info = utils.multiindex_to_frame(observations.indexes['obs_id'])
        return obs_info

    @staticmethod
    def _extract_state_information(
            state: xr.DataArray
    ) -> Tuple[pd.MultiIndex, xr.DataArray]:
        grid_index = state.indexes['grid']
        logger.debug('Got grid index')
        state_array = utils.index_to_array(grid_index.values)
        time_array = np.ones((state_array.shape[0], 1))
        time_array *= state.indexes['time'][0].timestamp()
        state_array = np.hstack((time_array, state_array))
        logger.debug('Got state id array')
        state_array = xr.DataArray(
            state_array,
            coords={
                'grid': np.arange(state_array.shape[0]),
                'id_names': np.arange(state_array.shape[1])
            },
            dims=['grid', 'id_names']
        )
        logger.debug('Transformed state id array into dataarray')
        return grid_index, state_array
