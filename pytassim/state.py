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
from typing import Union, Iterable, Dict, Tuple

# External modules
import xarray as xr
from xarray import register_dataarray_accessor

# Internal modules


logger = logging.getLogger(__name__)


class StateError(Exception):  # pragma: no cover
    """
    This error is an error if given state is not valid or if there is
    something strange with the state.
    """
    pass


@register_dataarray_accessor('state')
class ModelState(object):
    """
    This accessor extends an :py:class:`xarray.DataArray` by a ``state``
    property. This state property is used as utility class for an easier
    processing of model states for data assimilation purpose.

    Arguments
    ---------
    xr_da : :py:class:`xarray.DataArray`
        This model state accessor is registered to this array. The array should
        have (``variable``, ``time``, ``ensemble``, ``grid``) as coordinates,
        which are explained below. The values of the array have to be float.

        Coordinate explanation:
            ``var_name`` – str
                This coordinate allows to concatenate multiple variables into
                one single array. The coordinate should have :py:class:`str` as
                dtype.

            ``time`` – datetime like
                Different times can be concatenated into this coordinate. This
                coordinate is useful for data assimilation algorithms which
                allow assimilation of time dependent variables. The coordinate
                should be a datetime like dtype.

            ``ensemble`` – int
                This coordinate is used for ensemble based data assimilation
                algorithms. The ensemble members are numbered as
                :py:class:`int`. An integer value of 0 symbolizes a
                deterministic or control run. In some ensemble based algorithms,
                the ensemble coordinate is looped.

            ``grid``
                This coordinate is used to characterize the spatial position
                of one single value within the array. This coordinate can be a
                :py:class:`~pandas.MultiIndex`, where different coordinate
                components are stacked. In some algorithms the analysis is
                calculated for every grid point independently such that the
                algorithm loops over this coordinate.
    """
    def __init__(self, xr_da: xr.DataArray):
        self.array = xr_da

    def __str__(self):
        return 'ModelState({0}'.format(str(self.array))

    def __repr__(self):
        return 'ModelState'

    @property
    def _valid_dims(self) -> bool:
        """
        Checks if the dimension of this array are valid. The dimensions need
        to be in right order and have the right names as specified in
        documentation.

        Returns
        -------
        valid_dims : bool
            If the dimensions of this array have the right name and right order.
        """
        correct_dims = ('var_name', 'time', 'ensemble', 'grid')
        valid_dims = correct_dims == self.array.dims
        return valid_dims

    @property
    def valid(self) -> bool:
        """
        Checks if the array has the right form, coordinates and dimensions.

        Returns
        -------
        valid_array : bool
            If the given array is valid.
        """
        valid_array = self._valid_dims
        return valid_array

    def split_mean_perts(self,
                         dim: Union[str, Iterable[str]] = 'ensemble',
                         axis: Union[int, Iterable[int]] = None,
                         **kwargs: Dict) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Splits this :py:class:`~xarray.DataArray` into a mean array and a
        perturbations array by given dimension.

        Parameters
        ----------
        dim : str or sequence of str, optional
            The array is split over this dimension(s). Default is `'ensemble'`.
        axis : int or sequence of int, optional
            The array is split over this axis/axes. Only ``dim`` or ``axis`` can
            be supplied. Default is None.
        **kwargs : dict
            These additional keyword arguments are passed onto
            :py:meth:`~xarray.DataArray.mean` method of set array.

        Returns
        -------
        mean : :py:class:`~xarray.DataArray`
            Newly created array with averaged values. This array has the same
            coordinates as the original array.
        perts : :py:class:`~xarray.DataArray`
            Newly created array with perturbations as values. The perturbations
            are the difference of ``original array - mean``.
            This array has the same coordinates as the original array.
        """
        mean = self.array.mean(dim=dim, axis=axis, **kwargs)
        perts = self.array - mean
        return mean, perts
