#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 1/11/19
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2019}  {Tobias Sebastian Finn}
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
import inspect

# External modules

# Internal modules


logger = logging.getLogger(__name__)


class LossWrapper(object):
    """
    This loss wrapper wraps initialize :py:mod:`torch.nn` loss functions to make
    them available for autoencoder training.

    Parameters
    ----------
    loss : :py:class:`torch.nn.Module`
        This loss function should be an initialized loss function from
        :py:mod:`torch.nn`. At least, it should be callable with
        (input, target).
    """
    def __init__(self, loss):
        self._loss = None
        self.loss = loss

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, new_loss):
        if not hasattr(new_loss, '__call__'):
            raise TypeError('Given new loss is not callable!')
        self._loss = new_loss

    def recon_loss(self, recon_obs, observation, *args, **kwargs):
        """
        This reconstruction loss is used for the autoencoder to nudge the
        reconstructed observations to the real observations.

        Parameters
        ----------
        recon_obs : :py:torch:`torch.Tensor`
            The reconstruction loss is estimated based on these reconstructed
            observations.
        observation : :py:torch:`torch.Tensor`
            These observations are used as targets to esitmate the
            reconstruction loss. These observation should have the same tensor
            type as `recon_obs`.
        *args : iterable
            These additional arguments are not used.
        **kwargs : dict
            These additional keyword arguments are not used.

        Returns
        -------
        recon_loss : :py:class:`torch.Tensor`
            This reconstruction loss is estimated on given reconstructed
            observations and the trained network. This reconstruction loss has
            the same tensor type as `recon_obs`.

        Notes
        -----
        This reconstruction loss uses set loss function and passes given
        `recon_obs` as input and given `observation` as target.
        """
        recon_loss = self.loss(input=recon_obs, target=observation)
        return recon_loss

    def back_loss(self, analysis, prior=None, prior_ens=None, *args, **kwargs):
        """
        This background loss is used for the autoencoder to nudge the analysis
        to the prior.

        Parameters
        ----------
        analysis : :py:torch:`torch.Tensor`
            The background loss is estimated based on this estimated analysis.
        prior : :py:torch:`torch.Tensor` or None, optional
            This prior tensor can be used to estimate the background loss for
            analyses without ensembles. If this is None, then `prior_ens` is
            used. Default is None.
        prior_ens : :py:torch:`torch.Tensor` or None, optional
            This ensemble prior tensor can be used to estimate the background
            loss for analyses with ensembles. If `prior` is None, this prior is
            used. Default is None.
        *args : iterable
            These additional arguments are not used.
        **kwargs : dict
            These additional keyword arguments are not used.

        Returns
        -------
        back : :py:class:`torch.Tensor`
            This is the estimated background loss based on given analysis and
            prior. This background loss has the same tensor type as `analysis`.

        Raises
        ------
        ValueError:
            If neither `prior` or `prior_ens` is given as argument.

        Notes
        -----
        This uses set loss function to estimate the background loss. Given
        analysis is used as input, while given `prior` or `prior_ens` is used
        as target. Either `prior` or `prior_ens` has to be given as argument!
        """
        if prior is not None:
            back_loss = self.loss(input=analysis, target=prior)
        elif prior_ens is not None:
            back_loss = self.loss(input=analysis, target=prior_ens)
        else:
            raise ValueError(
                'Either a determinstic `prior` or an ensemble `prior_ens` have '
                'to be given for background loss!')
        return back_loss

