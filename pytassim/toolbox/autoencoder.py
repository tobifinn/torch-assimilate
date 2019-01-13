#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 1/10/19
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
import itertools

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


class Autoencoder(object):
    """
    This module is designed to train an inference network for data
    assimilation based on an autoencoder. This module contains methods to train
    the inference network and observation operator (can be fixed) jointly, based
    on specified loss functions. This module can be also used for assimilation
    purpose in a predefined way. To freeze a network, `requires_grad` in its
    parameters has to be set to False. This is especially useful if a physical
    plausible observation operator is used.

    Parameters
    ----------
    inference_net : :py:class:`torch.nn.Module`
        This network is used for inference and combines given inputs into an
        analysis. The main purpose of this autoencoder is to train this given
        network. In its forward method, the arguments for the inference network
        are `observation`, `prior`, `prior_ensemble`, `noise`. If some arguments
        are not used, they can be set to a default value of None. The forward
        method has to return a single output, the analysis. This inference
        network needs also an `assimilate` method, where `in_state` (the prior),
        `obs` (the observations) and `obs_cov` (the observation covariance) are
        given as 1D (prior and obs) or 2D (covariance) tensors. This method
        should return only the analysis as output.
    obs_operator : :py:class:`torch.nn.Module`
        This observation operator network is used to translate from model space
        into observation space. The forward method of the observation operator
        takes a model state and returns pseudo observations.

    Attributes
    ----------
    inference_net : :py:class:`torch.nn.Module`
        This network is used for inference and combines given inputs into an
        analysis. The main purpose of this autoencoder is to train this given
        network. This network will be also used for assimilation.
    obs_operator : :py:class:`torch.nn.Module`
        This observation operator network is used to translate from model space
        into observation space.
    recon_loss : child of :py:class:`pytassim.toolbox.loss.BaseLoss` or
    child of :py:class:`pytassim.toolbox.discriminator.standard.StandardDisc`
        This reconstruction loss instance should have a `recon_loss` method,
        which compares the reconstructed observations and the real observations.
    back_loss : child of :py:class:`pytassim.toolbox.loss.BaseLoss` or
    child of :py:class:`pytassim.toolbox.discriminator.standard.StandardDisc`
        This background loss instance should have a `back_loss` method, which
        compared the analysis with another given or estimated quantities.
    optimizer : child of :py:class:`torch.optim.Optimizer`
        This optimizer is used to update the trainable parameters of this
        autoencoder.
    """
    def __init__(self, inference_net, obs_operator, ):
        self.inference_net = inference_net
        self.obs_operator = obs_operator
        self.recon_loss = None
        self.back_loss = None
        self.optimizer = None

    @property
    def trainable_params(self):
        """
        The trainable parameters of this autoencoder. The parameters are
        concatenated from the inference network and observation operator.
        Parameters with no required gradient are skipped.

        Returns
        -------
        trainable_params : list of :py:class:`torch.nn.Parameter`
            List of trainable parameters with needed gradient from inference_net
            and observation operator.
        """
        parameters = itertools.chain(
            self.inference_net.parameters(), self.obs_operator.parameters()
        )
        trainable_params = [p for p in parameters if p.requires_grad]
        return trainable_params

    def check_trainable(self):
        """
        Check if this autoencoder is trainable and `recon_loss`, `back_loss` and
        `optimizer` are set to the right values.
        """
        if not hasattr(self.recon_loss, 'recon_loss'):
            raise TypeError(
                'Set recon_loss is not a valid reconstruction loss!'
            )
        if not hasattr(self.back_loss, 'back_loss'):
            raise TypeError(
                'Set back_loss is not a valid background loss!'
            )
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError(
                'Set optimizer is not a valid pytorch optimizer!'
            )
        if not self.trainable_params:
            raise ValueError(
                'There are no trainable parameters!'
            )

    def forward(self, observation, prior=None, prior_ensemble=None,
                noise=None):
        """
        This uses the inference network to infer an analysis and reconstructs
        the observations with set observation operator.

        Parameters
        ----------
        observation : :py:class:`torch.Tensor`
            This is the observations tensor, the inference is based on this
            tensor.
        prior : :py:class:`torch.Tensor` or None, optional
            This prior tensor can be used for a deterministic or stochastic
            inference. If no deterministic prior should be used then this has
            to be None. Default is None.
        prior_ensemble : :py:class:`torch.Tensor` or None, optional
            This prior ensemble can be given to update the whole ensemble or to
            update given prior by ensemble statistics. If no ensemble should be
            used then this has to be None. Default is None.
        noise : :py:class:`torch.Tensor` or None, optional
            This noise input can be used to construct a stochastic inference.
            Noise input is normally used to artifically inflate an ensemble.
            If no noise input should be used then this has to be None. Default
            is None.

        Returns
        -------
        analysis : :py:class:`torch.Tensor`
            This analysis is estimated with the inference network and is based
            on given input arguments.
        recon_obs : :py:class:`torch.Tensor`
            These reconstructed observations are estimated with the observation
            operator based on estimated analysis.
        """
        analysis = self.inference_net(
            observation=observation, prior=prior, prior_ensemble=prior_ensemble,
            noise=noise
        )
        recon_obs = self.obs_operator(analysis)
        return analysis, recon_obs

    def _get_train_losses(self, observation, prior=None, prior_ensemble=None,
                          noise=None):
        analysis, recon_obs = self.forward(
            observation=observation, prior=prior, prior_ensemble=prior_ensemble,
            noise=noise
        )

        back_loss = self.back_loss.back_loss(
            analysis, observation=observation, prior=prior,
            prior_ensemble=prior_ensemble, noise=noise
        )

        recon_loss = self.recon_loss.recon_loss(
            recon_obs, observation=observation, prior=prior,
            prior_ensemble=prior_ensemble, noise=noise
        )

        total_loss = back_loss + recon_loss
        return total_loss, back_loss, recon_loss

    def train(self, observation, prior=None, prior_ensemble=None, noise=None):
        """
        This method is used to train this autoencoder with given parameters and
        set loss functions.

        Parameters
        ----------
        observation : :py:class:`torch.Tensor`
            This is the observation, which is mandatory and used to estimate the
            analysis.
        prior : :py:class:`torch.Tensor` or None, optional
            This is the prior, which can be used as deterministic prior
            information about the latent state. If this prior is None, it will
            not be used by the inference network. Default is None.
        prior_ensemble : :py:class:`torch.Tensor` or None, optional
            This is the ensemble prior, which can be used as stochastic prior
            information about the latent state. If this prior is None, it will
            not be used by the inference network. Default is None.
        noise : :py:class:`torch.Tensor` or None, optional
            This random noise can be used to artifically inflate the resulting
            analysis or to translate a deterministic network into a stochastic
            one. If this noise is None, it will be not used by the inference
            network. Default is None.

        Returns
        -------
        tot_loss : :py:class:`torch.Tensor`
            This is the total loss of this autoencoder, which was used to train
            the models.
        back_loss : :py:class:`torch.Tensor`
            This is the background loss of this autoencoder, which compares
            analysis with prior and tries to draw the inference network to the
            prior.
        recon_loss : :py:class:`torch.Tensor`
            This is the reconstruction loss of this autoencoder, which compares
            the reconstructed observation based on estimated analysis and tries
            to draw this reconstructed observation to given observation.

        Warnings
        --------
        To train this autoencoder, loss functions in `back_loss` and
        `recon_loss` and an optimizer has to be set!
        """
        self.check_trainable()

        self.inference_net.train()
        self.obs_operator.train()
        self.optimizer.zero_grad()

        total_loss, back_loss, recon_loss = self._get_train_losses(
            observation=observation, prior=prior, prior_ensemble=prior_ensemble,
            noise=noise
        )
        back_loss.backward(retain_graph=True)
        recon_loss.backward()

        self.optimizer.step()
        return total_loss, back_loss, recon_loss

    def eval(self, observation, prior=None, prior_ensemble=None, noise=None):
        """
        Evaluate this autoencoder with given observations and other arguments.

        Parameters
        ----------
        observation : :py:class:`torch.Tensor`
            This is the observation, which is mandatory and used to estimate the
            analysis.
        prior : :py:class:`torch.Tensor` or None, optional
            This is the prior, which can be used as deterministic prior
            information about the latent state. If this prior is None, it will
            not be used by the inference network. Default is None.
        prior_ensemble : :py:class:`torch.Tensor` or None, optional
            This is the ensemble prior, which can be used as stochastic prior
            information about the latent state. If this prior is None, it will
            not be used by the inference network. Default is None.
        noise : :py:class:`torch.Tensor` or None, optional
            This random noise can be used to artifically inflate the resulting
            analysis or to translate a deterministic network into a stochastic
            one. If this noise is None, it will be not used by the inference
            network. Default is None.

        Returns
        -------
        tot_loss : :py:class:`torch.Tensor`
            This is the total loss of this autoencoder, which was used to train
            the models.
        back_loss : :py:class:`torch.Tensor`
            This is the background loss of this autoencoder, which compares
            analysis with prior and tries to draw the inference network to the
            prior.
        recon_loss : :py:class:`torch.Tensor`
            This is the reconstruction loss of this autoencoder, which compares
            the reconstructed observation based on estimated analysis and tries
            to draw this reconstructed observation to given observation.

        Warnings
        --------
        To evaluate this autoencoder, `back_loss` and `recon_loss` has to be
        set!
        """
        self.inference_net.eval()
        self.obs_operator.eval()

        total_loss, back_loss, recon_loss = self._get_train_losses(
            observation=observation, prior=prior, prior_ensemble=prior_ensemble,
            noise=noise
        )
        return total_loss, back_loss, recon_loss
