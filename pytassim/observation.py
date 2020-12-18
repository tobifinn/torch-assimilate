#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 14.03.18
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
import types
from inspect import signature

# External modules
import xarray as xr
from xarray import register_dataset_accessor
import dask.array as da
import numpy as np

# Internal modules
from .utilities import lazy_property


logger = logging.getLogger(__name__)


class ObservationError(Exception):  # pragma: no cover
    """
    This error is an error if given observation is not valid or if there is
    something strange with the observations.
    """
    pass


@register_dataset_accessor('obs')
class Observation(object):
    """
    This class represents an observation subset. Within an observation subset,
    the observations can be spatial correlated. Different observation subsets
    are per definition uncorrelated. This class is registered under
    :py:attr:`xarray.Dataset.obs`, where the here defined attributes and methods
    can be accessed. The assimilation algorithms will use the abstract method
    :py:meth:`xarray.Dataset.obs.operator` to convert the given model field into
    an observation equivalent. Thus, the `operator` method needs to be
    overwritten for every observation subset independently.

    Arguments
    ---------
    xr_ds : :py:class:`~xarray.Dataset`
        This :py:class:`~xarray.Dataset` is used for the observation operator.
        The dataset needs two variables:

            observations
                (time, obs_grid_1), the actual observation values

            covariance
                (obs_grid_1) or (obs_grid_1, obs_grid_2), the covariance between
                different observations. If this is a vector, then it is assumed
                that the observations are uncorrelated and only the variances
                are witihn this array.

        ``obs_grid_1`` and ``obs_grid_2`` are the same, but due to internals of
        xarray they are saved under different coordinates. It is possible to
        define different observation times within the `time` coordinate of the
        given :py:class:`~xarray.Dataset`.

    Warnings
    --------
    **To use this observation subset, you need to overwrite the observation
    operator**
    """
    def __init__(self, xr_ds: xr.Dataset):
        self.ds = xr_ds
        self._r_cinv = None

    def __str__(self):
        return 'Obs dataset ({0})'.format(str(self.ds))

    def __repr__(self):
        return 'Observation'

    @property
    def correlated(self) -> bool:
        """
        Checks if the observations are correlated based on the dimensions of
        given dataset.

        Returns
        -------
        correlated : bool
            If the observations are correlated
        """
        correlated = 'obs_grid_2' in self.ds['covariance'].dims
        return correlated

    @property
    def _valid_dims(self) -> bool:
        """
        Checks if ``time``, ``obs_grid_1`` are available within the
        :py:class:`~xarray.Dataset` and if ``time``, ``obs_grid_1`` and
        ``obs_grid_2`` are the only dimensions.

        Returns
        -------
        valid_dims : bool
            If the dimensions are available.
        """
        necessary_dims = [
            'time', 'obs_grid_1'
        ]
        keys_avail = all(
            True if d in tuple(self.ds.dims.keys()) else False
            for d in necessary_dims
        )

        all_dims = necessary_dims + ['obs_grid_2']
        no_auxiliary = all(
            True if d in all_dims else False for d in tuple(self.ds.dims.keys())
        )

        valid_dims = keys_avail and no_auxiliary
        return valid_dims

    @property
    def _valid_obs(self) -> bool:
        """
        Checks if dimensions of the ``observation``
        :py:class:`~xarray.DataArray` within the set dataset are valid.

        Returns
        -------
        valid_obs : bool
            If the ``observation`` :py:class:`~xarray.DataArray` is valid.
        """
        valid_dims = ('time', 'obs_grid_1')
        valid_obs = valid_dims == self.ds['observations'].dims[-2:]
        return valid_obs

    @property
    def _valid_cov_uncorr(self) -> bool:
        """
        Checks if shape and dimensions of the uncorrelated ``covariance``
        :py:class:`~xarray.DataArray` within the set dataset are valid.

        Returns
        -------
        valid_cov : bool
            If the ``covariance`` :py:class:`~xarray.DataArray` is valid.
        """
        obs_grid_len = self.ds['observations'].shape[-1]
        if 'time' in self.ds['covariance'].dims:
            dim_order = ('time', 'obs_grid_1')
            valid_shape = (self.ds['time'].shape[0], obs_grid_len)
        else:
            dim_order = ('obs_grid_1',)
            valid_shape = (obs_grid_len,)

        checked_dims = dim_order == self.ds['covariance'].dims
        checked_shape = valid_shape == self.ds['covariance'].shape
        valid_cov = checked_dims and checked_shape
        return valid_cov

    @property
    def _valid_cov_corr(self) -> bool:
        """
        Checks if shape and dimensions of the correlated ``covariance``
        :py:class:`~xarray.DataArray` within the set dataset are valid.

        Returns
        -------
        valid_cov : bool
            If the ``covariance`` :py:class:`~xarray.DataArray` is valid.
        """
        obs_grid_len = self.ds['observations'].shape[-1]

        if 'time' in self.ds['covariance'].dims:
            dim_order = ('time', 'obs_grid_1', 'obs_grid_2')
            valid_shape = (self.ds['time'].shape[0], obs_grid_len, obs_grid_len)
        else:
            dim_order = ('obs_grid_1', 'obs_grid_2')
            valid_shape = (obs_grid_len, obs_grid_len)

        checked_dims = dim_order == self.ds['covariance'].dims
        checked_shape = valid_shape == self.ds['covariance'].shape

        try:
            checked_coord_values = self.ds['obs_grid_1'].to_index().equals(
                self.ds['obs_grid_2'].to_index()
            )
        except KeyError:
            checked_coord_values = False

        valid_cov = checked_dims and checked_shape and checked_coord_values
        return valid_cov

    @property
    def _valid_arrays(self) -> bool:
        """
        Checks if ``observations`` and ``covariance``
        :py:class:`~xarray.DataArray`s within the set dataset are valid.

        Returns
        -------
        valid_arrays : bool
            If the two :py:class:`~xarray.DataArray`s are valid.
        """
        try:
            if self.correlated:
                valid_array = self._valid_obs and self._valid_cov_corr
            else:
                valid_array = self._valid_obs and self._valid_cov_uncorr
        except KeyError:
            valid_array = False
        return valid_array

    @property
    def valid(self) -> bool:
        """
        Checks if set :py:class:`~xarray.Dataset` is valid.

        Returns
        -------
        valid_ds : bool
            If set :py:class:`~xarray.Dataset` is valid.
        """
        valid_ds = False
        if self._valid_dims and self._valid_arrays:
            valid_ds = True
        return valid_ds

    @lazy_property('r_cinv')
    def _uncorr_chol_inverse(self) -> np.ndarray:
        cov_square_root = da.sqrt(self.ds['covariance'])
        cov_chol_inv = 1 / cov_square_root
        return cov_chol_inv

    @staticmethod
    def _single_corr_chol_inv(array: xr.DataArray) -> xr.DataArray:
        chol = np.linalg.cholesky(array.data)
        chol_inv = np.linalg.inv(chol)
        chol_inv = array.copy(data=chol_inv)
        return chol_inv

    @lazy_property('r_cinv')
    def _corr_chol_inverse(self):
        if 'time' in self.ds['covariance'].dims:
            cov_chol_inv = []
            for t in self.ds.indexes['time']:
                curr_cov = self.ds['covariance'].sel(time=t)
                curr_chol_inv = self._single_corr_chol_inv(curr_cov)
                cov_chol_inv.append(curr_chol_inv)
            cov_chol_inv = xr.concat(cov_chol_inv, dim='time')
        else:
            chol_decomp = np.linalg.cholesky(self.ds['covariance'].data)
            cov_chol_inv = np.linalg.inv(chol_decomp)
            cov_chol_inv = self.ds['covariance'].copy(data=cov_chol_inv)
        return cov_chol_inv

    def _corr_normalize(self, value):
        normalized = xr.dot(value, self._corr_chol_inverse, dims='obs_grid_1')
        normalized = normalized.rename({'obs_grid_2': 'obs_grid_1'})
        return normalized

    def _uncorr_normalize(self, value):
        normalized = value * self._uncorr_chol_inverse
        return normalized

    def mul_rcinv(self, value):
        if self.correlated:
            return self._corr_normalize(value)
        else:
            return self._uncorr_normalize(value)

    @staticmethod
    def operator(obs_ds: xr.Dataset, state: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError('No observation operator is set!')
