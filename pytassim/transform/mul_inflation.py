#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 19.01.21
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union, Iterable, Tuple

# External modules
import numpy as np
import xarray as xr

# Internal modules
from .base import BaseTransformer


logger = logging.getLogger(__name__)


class MultiplicativeInflation(BaseTransformer):
    def __init__(self, inf_factor: float = 1.0):
        super().__init__()
        self.inf_factor = inf_factor

    def _inflate_array(self, array_to_inflate: xr.DataArray) -> xr.DataArray:
        array_mean = array_to_inflate.mean('ensemble')
        array_perts = array_to_inflate - array_mean
        inflated_perts = np.sqrt(self.inf_factor) * array_perts
        inflated_array = inflated_perts + array_mean
        return inflated_array

    def pre(
            self,
            background: xr.DataArray,
            observations: Iterable[xr.Dataset],
            first_guess: Union[xr.DataArray, None]
    ) -> Tuple[
        xr.DataArray,
        Union[Iterable[xr.Dataset], xr.Dataset],
        Union[xr.DataArray, None]
    ]:
        inflated_background = self._inflate_array(background)
        if isinstance(first_guess, xr.DataArray):
            inflated_first_guess = self._inflate_array(first_guess)
        else:
            inflated_first_guess = first_guess
        return inflated_background, observations, inflated_first_guess

    def post(
            self,
            analysis: xr.DataArray,
            background: xr.DataArray,
            observations: Iterable[xr.Dataset],
            first_guess: Union[xr.DataArray, None]
    ) -> xr.DataArray:
        inflated_analysis = self._inflate_array(analysis)
        return inflated_analysis
