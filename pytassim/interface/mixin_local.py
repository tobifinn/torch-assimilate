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

# Internal modules
import pytassim.utilities.pandas as utils


logger = logging.getLogger(__name__)


class DomainLocalizedMixin(object):
    def __init__(self):
        self.localization = None

    @staticmethod
    def _extract_obs_information(observations: xr.DataArray) -> pd.DataFrame:
        obs_info = utils.multiindex_to_frame(observations.indexes['obs_id'])
        return obs_info

    @staticmethod
    def _extract_state_information(
            state: xr.DataArray
    ) -> Tuple[pd.MultiIndex, xr.DataArray]:
        state_id = state.assign_coords(
            time=utils.dtindex_to_total_seconds(state.indexes['time'])
        )
        state_id = state_id.state.stack(state_id=['time', 'grid'])
        state_index = state_id.indexes['state_id']
        state_array = utils.index_to_array(state_id.indexes['state_id'])
        state_array = xr.DataArray(
            state_array,
            coords={
                'state_id': np.arange(state_array.shape[0]),
                'id_names': np.arange(state_array.shape[1])
            },
            dims=['state_id', 'id_names']
        )
        return state_index, state_array

    def localization_decorator(self, func):
        def wrapper(grid_info, *args, obs_info=None, **kwargs):
            if self.localization is not None:
                luse, lweights = self.localization.localize_obs(
                    grid_info, obs_info
                )
                lweights = lweights[luse]
                args = [arg[..., luse]*lweights for arg in args]
            return func(*args, **kwargs)
        return wrapper
