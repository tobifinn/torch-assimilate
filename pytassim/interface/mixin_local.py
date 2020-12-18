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

# External modules
import xarray as xr
import numpy as np

# Internal modules
from .utils import index_to_array


logger = logging.getLogger(__name__)


class DomainLocalizedMixin(object):
    @staticmethod
    def _extract_obs_information(observations: xr.DataArray) -> np.ndarray:
        obs_infos = index_to_array(observations['obs_id'].data)
        return obs_infos

    def _extract_state_information(self, state):
        pass

    def apply_dlocal_ufunc(self):
        pass
