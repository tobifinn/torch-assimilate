#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 26.03.18
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
from copy import deepcopy
from typing import Union, Iterable

# External modules
import torch.nn

import xarray as xr
import pandas as pd

# Internal modules
from ..base import BaseAssimilation
from pytassim.transform import BaseTransformer


logger = logging.getLogger(__name__)


class NeuralAssimilation(BaseAssimilation):
    """
    NeuralAssimilation is a class to assimilate observations into a state with
    neural networks. This class supports PyTorch natively. The ``assimilate``
    method of a given :py:class:`torch.nn.Module` is used to to assimilate the
    given state. No observation operator is needed for given observations.

    Parameters
    ----------
    model : child of :py:class:`torch.nn.Module`
        This model is used to assimilate given observations into given state.
        The model needs an ``assimilate`` method, where state, flattened
        observations and flattened observation covariance is given. This model
        is transferred to specified device. Attention, a new clone of this
        model is created.
    smoother : bool, optional
        This bool indicates if given `state` and `observations` should be
        localized to given `analysis_time`. If True, full state and all
        observations are passed to neural network. If False, localized state and
        nearest observations are only used. Default is True.
    gpu : bool, optional
        This boolean indicates if the assimilation should be done on GPU (True)
        or CPU (False). Default value is False, indicating computations on CPU.
        This flag transfers also given model to CPU or GPU.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            smoother: bool = False, gpu: bool = False,
            pre_transform: Union[None, Iterable[BaseTransformer]] = None,
            post_transform: Union[None, Iterable[BaseTransformer]] = None
    ):
        super().__init__(smoother=smoother, gpu=gpu,
                         pre_transform=pre_transform,
                         post_transform=post_transform)
        self._model = None
        self.model = model

    def __str__(self) -> str:
        return 'NeuralAssimilation'

    def __repr__(self) -> str:
        return 'NeuralAssimilation'

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, new_model: torch.nn.Module):
        if not hasattr(new_model, 'assimilate'):
            raise TypeError('Given model is not a valid assimilation model!')
        cloned_model = deepcopy(new_model)
        cloned_model = cloned_model.type(self.dtype)
        if self.gpu:
            self._model = cloned_model.cuda()
        else:
            self._model = cloned_model.cpu()

    def update_state(
            self,
            state: xr.DataArray,
            observations: Union[xr.Dataset, Iterable[xr.Dataset]],
            pseudo_state: xr.DataArray,
            analysis_time: pd.Timestamp
    ) -> xr.DataArray:
        """
        This method updates given `state` with given `observations`. This method
        stacks the observations and covariance together. The state values and
        observations are localized to given `analysis_time`, if ``smoother`` is
        set to False. In an additional step, the prepared states are
        transferred to :py:class:`torch.tensor`, on either CPU or GPU, depending
        on set `gpu` flag. :py:meth:`self.model.assimilate` is called for
        assimilation with prepared state, observations and observation
        covariance. Estimated analysis is converted into a
        :py:class:`xarray.DataArray`, which is returned.

        Parameters
        ----------
        state : :py:class:`xarray.DataArray`
            This state is localized in time and passed to
            :py:meth:`self.model.assimilate` as first argument. It is
            further updated by this assimilation algorithm and given
            ``observation``. This :py:class:`~xarray.DataArray` should have
            four coordinates, which are specified in
            :py:class:`pytassim.state.ModelState`.
        observations : :py:class:`xarray.Dataset` or \
        iterable(:py:class:`xarray.Dataset`)
            These observations are used to update given state. An iterable of
            many :py:class:`xarray.Dataset` can be used to assimilate different
            variables. For the observation state, these observations are
            stacked such that the observation state contains all observations.
        analysis_time : :py:class:`datetime.datetime`
            This analysis time determines at which point the state is updated.
            This argument is not used if `smoother` is set to True.

        Returns
        -------
        analysis : :py:class:`xarray.DataArray`
            The analysed state based on given state and observations. The
            analysis has same coordinates as given ``state``. If filtering mode
            is on, then the time axis has only one element.
        """
        logger.info('####### Neural network assimilation #######')
        logger.info('Preparing observations')
        obs_state, _ = self._prepare_obs(observations)
        logger.info('Transferring data to torch')
        prepared_torch = self._states_to_torch(state.values, obs_state,
                                               pseudo_state.values)
        logger.info('Assimilating observations with model')
        torch_analysis = self.model.assimilate(*prepared_torch)
        logger.info('Gathering analysis')
        if self.gpu:
            torch_analysis = torch_analysis.cpu()
        analysis = state.copy(deep=True, data=torch_analysis.numpy())
        logger.info('Finished with analysis creation')
        return analysis
