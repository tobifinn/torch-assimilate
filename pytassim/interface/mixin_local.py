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
from .wrapper import wrapper_localization


logger = logging.getLogger(__name__)


class DomainLocalizedMixin(object):
    @property
    def localized_module(self):
        wrapped_module = wrapper_localization(
            module=self.module,
            localization=self.localization
        )
        return wrapped_module

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
