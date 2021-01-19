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
    """
    This transformer implements multiplicative inflation with a given
    inflation factor.
    The ensemble perturbations are multiplied by the square-root of this
    inflation factor such that the corresponding covariance is inflated by
    given inflation factor.

    Parameters
    ----------
    inf_factor : float, optional
        The ensemble perturbations are multiplied with the square-root if
        this inflation factor.
        There is no testing if the inflation factor is valid or not!
    """
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
            first_guess: Union[xr.DataArray, None] = None
    ) -> Tuple[
        xr.DataArray,
        Union[Iterable[xr.Dataset], xr.Dataset],
        Union[xr.DataArray, None]
    ]:
        """
        Inflate a given background array and a possibly given first guess
        array by the square-root of the set inflation factor.
        This method corresponds to prior multiplicative inflation.

        Parameters
        ----------
        background : xarray.DataArray
            The ensemble perturbations of this background array will be
            inflated.
        observations : Iterable[xarray.Dataset]
            These observations are not influenced by the inflation.
        first_guess : xarray.DataArray or None, optional
            If a first guess is given, the ensemble perturbations of this
            first guess will be inflated.
            No first guess is given per default.

        Returns
        -------
        inflated_background : xr.DataArray
            The inflated background.
        observations : Iterable[xarray.Dataset]
            The untouched observations.
        inflated_first_guess : xarray.DataArray or None
            If this is a xarray.DataArray, the first guess was inflated.
        """
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
        """
        Inflate a given analysis array by the square-root of the set inflation
        factor.
        This method corresponds to posterior multiplicative inflation.

        Parameters
        ----------
        analysis : xr.DataArray
            The ensemble perturbations of this analysis array will be
            inflated by the square-root of the set inflation factor.
        background : xr.DataArray
            This background array is not used.
        observations : Iterable[xarray.Dataset]
            These observational datasets are not used.
        first_guess : xarray.DataArray
            This first guess array is not used.

        Returns
        -------
        inflated_analysis : xarray.DataArray
            The analysis array with the inflated ensemble perturbations.
        """
        inflated_analysis = self._inflate_array(analysis)
        return inflated_analysis
