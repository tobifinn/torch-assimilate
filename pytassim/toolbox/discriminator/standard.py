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

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


class StandardDisc(object):
    """
    Standard discriminator for generative adversarial networks. This
    discriminator takes input data and returns processed data based on specified
    neural network. This discriminator can be also used as base discriminator to
    implement other types of discriminators for generative adversarial networks.

    Parameters
    ----------
    net : :py:class:`pytorch.nn.Module`
        This network is used for this discriminator. This network is also
        updated during training. This network should take

    Attributes
    ----------
    net : :py:class:`pytorch.nn.Module`
        This network is used for this discriminator. This network is also
        updated during training.
    loss_func : callable
        This loss function should be an initialized loss function from
        :py:mod:`torch.nn`.
    """
    def __init__(self, net,):
        self.net = net
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.optimizer = None
        self.grad_optim = True

    @property
    def trainable_params(self):
        """
        List of trainable parameters from set discriminator network. Trainable
        parameters have a required gradient.

        Returns
        -------
        trainable_params : list of :py:class:`torch.nn.Parameter`
            Trainable parameters from this discriminator.
        """
        trainable_params = [p for p in self.net.parameters() if p.requires_grad]
        return trainable_params

    def check_trainable(self):
        """
        Check if this discriminator is trainable. The discriminator is trainable
        if a valid loss function and a valid optimizer is set. It is also
        checked if there are trainable parameters at all.
        """
        if not hasattr(self.loss_func, '__call__'):
            raise TypeError('Set loss function is not a valid callable '
                            'loss function!')

        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError('Set optimizer is not a valid torch optimizer!')
        if not self.trainable_params:
            raise ValueError('This discriminator has no trainable parameters '
                             'and cannot be trained!')

    @staticmethod
    def get_targets(batch_size, fill_val=1.0, tensor_type=None):
        """
        This method returns targets for discriminator loss.

        Parameters
        ----------
        batch_size : int
            This is the batch size of the targets. The resulting torch tensor
            will have this batch size as first dimension.
        fill_val : any, optional
            This/These fill value(s) is/are used to fill in the torch tensor.
            The default value is 1.0.
        tensor_type : any or None, optional
            This tensor type will be passed to :py:meth:`torch.Tensor.to` as
            first argument, if this is not None. This tensor type can be used
            to cast the targets onto another device or into another dtype.
            Default is None.

        Returns
        -------
        targets : :py:class:`torch.Tensor`
            This target tensor will have as dimension `batch_size * 1` and
            requires not gradient. This tensor is filled with given fill
            value(s) and casted into given `tensor_type`.
        """
        targets = torch.full((batch_size, 1), fill_val,
                             requires_grad=False)
        if tensor_type is not None:
            targets = targets.to(tensor_type)
        return targets

    def disc_loss(self, in_data, labels):
        """
        This discriminator loss takes the in_data and labels and returns the
        discriminator loss.

        Parameters
        ----------
        in_data : :py:class:`torch.Tensor`
            This is the discriminator output, outputted by `forward`. In this
            discriminator, this are logits, which are translated into
            probabilities.
        labels : :py:class:`torch.Tensor`
            These are the labels, which are used to determine the loss function
            value for this discriminator. In this discriminator, this is a
            probability label. These labels should have the same tensor type as
            `in_data`.

        Returns
        -------
        loss : :py:class:`torch.Tensor`
            The loss for given input data and labels. In this discriminator, it
            is the cross-entropy between estimated probability and labels. This
            loss has the same tensor type as given `in_data`.
        """
        loss = self.loss_func(in_data, labels)
        return loss

    def forward(self, *args, **kwargs):
        """
        This method calls set net to generate a discriminator critic output,
        which can be compared to targets.

        Parameters
        ----------
        *args : iterable of :py:class:`torch.Tensor`
            These variable length arguments are passed to set discriminator
            network.
        **kwargs : dict(str, :py:class:`torch.Tensor`)
            These variable length keyword arguments are passed to set
            discriminator network additional keyword arguments.

        Returns
        -------
        out_data : :py:class:`torch.Tensor`
            The estimated discriminator critic, which can be compared to
            targets.
        """
        out_data = self.net(*args, **kwargs)
        return out_data

    def _get_train_losses(self, real_data, fake_data, *args, **kwargs):
        batch_size = real_data.size()[0]

        real_critic = self.forward(real_data, *args, **kwargs)
        real_labels = self.get_targets(batch_size, 1.0, real_data)
        real_loss = self.disc_loss(real_critic, real_labels)

        fake_critic = self.forward(fake_data, *args, **kwargs)
        fake_labels = self.get_targets(batch_size, 0.0, real_data)
        fake_loss = self.disc_loss(fake_critic, fake_labels)

        total_loss = real_loss + fake_loss
        return total_loss, real_loss, fake_loss

    def set_grad(self, real_data, fake_data, *args, **kwargs):
        self.check_trainable()

        self.net.train()
        self.optimizer.zero_grad()

        total_loss, real_loss, fake_loss = self._get_train_losses(
            real_data, fake_data, *args, **kwargs
        )

        real_loss.backward(retain_graph=True)
        fake_loss.backward()
        return total_loss, real_loss, fake_loss

    def train(self, real_data, fake_data, closure=None, *args, **kwargs):
        """
        Train this discriminator on given real data and fake data.

        Parameters
        ----------
        real_data : :py:class:`torch.Tensor`
            This tensor is used as real data input for this discriminator. The
            first dimension is also used as batch size, to generate the targets.
        fake_data : :py:class:`torch.Tensor`
            This tensor is used as fake data input to train this discriminator.
            This fake data should have the same tensor type as the real data.
        closure : callable or None, optional
            A closure that reevaluates the model and returns the loss. Optional
            for most optimizers. If None, it will not be used during
            optimization. Default is None.
        *args : iterable(any), optional
            This variable length list of tensors is used as additional arguments
            to train the network. These additional arguments are passed to the
            forward method of the network.
        **kwargs : dict(str, any), optional
            These additional keyword arguments are used as additional arguments
            to train the network. These additional keyword arguments are passed
            to the forward method of the network.

        Returns
        -------
        tot_loss : :py:class:`torch.Tensor`
            This is the total loss of this discriminator and is a combination of
            the fake and real loss. This total loss tensor has the same tensor
            type as given `real_data`. Normally, this is
            :math:`real_loss` + `fake_loss`.
        real_loss : :py:class:`torch.Tensor`
            This is the discriminator loss of the real data. This real data loss
            has the same tensor type as given `real_data`.
        fake_loss : :py:class:`torch.Tensor`
            This is the discriminator loss of the fake data. This fake data loss
            has the same tensor type as given `real_data`.

        Warnings
        --------
        To train this discriminator, a valid loss function and optimizer has to
        be set and also this discriminator needs trainable parameters.
        """
        if self.grad_optim:
            total_loss, real_loss, fake_loss = self.set_grad(
                real_data, fake_data, *args, **kwargs
            )
        if closure is None:
            self.optimizer.step()
        else:
            total_loss, real_loss, fake_loss = self.optimizer.step(closure)
        return total_loss, real_loss, fake_loss

    def eval(self, real_data, fake_data, *args, **kwargs):
        """
        Evaluate this discriminator on given real data and fake data.

        Parameters
        ----------
        real_data : :py:class:`torch.Tensor`
            This tensor is used as real data input for this discriminator. The
            first dimension is also used as batch size, to generate the targets.
        fake_data : :py:class:`torch.Tensor`
            This tensor is used as fake data input to evaluate this
            discriminator. This fake data should have the same tensor type as
            the real data.
        *args : iterable(any), optional
            This variable length list of tensors is used as additional arguments
            to evaluate the network. These additional arguments are passed to
            the forward method of the network.
        **kwargs : dict(str, any), optional
            These additional keyword arguments are used as additional arguments
            to evaluate the network. These additional keyword arguments are
            passed to the forward method of the network.

        Returns
        -------
        tot_loss : :py:class:`torch.Tensor`
            This is the total loss of this discriminator and is a combination of
            the fake and real loss. This total loss tensor has the same tensor
            type as given `real_data`. Normally, this is
            :math:`real_loss` + `fake_loss`.
        real_loss : :py:class:`torch.Tensor`
            This is the discriminator loss of the real data. This real data loss
            has the same tensor type as given `real_data`.
        fake_loss : :py:class:`torch.Tensor`
            This is the discriminator loss of the fake data. This fake data loss
            has the same tensor type as given `real_data`.
        """
        self.net.eval()
        total_loss, real_loss, fake_loss = self._get_train_losses(
            real_data, fake_data, *args, **kwargs
        )
        return total_loss, real_loss, fake_loss

    def gen_loss(self, fake_data, *args, **kwargs):
        """
        This loss can be used to train a generator based on critic values of
        this discriminator. In this discriminator, it is the cross-entropy of
        the probability that fake data is discriminated as real data.

        Parameters
        ----------
        fake_data : :py:class:`torch.Tensor`
            These fake data are used to estimate the critics value of this
            discriminator. The first dimension of this tensor is used as batch
            size.
        *args : iterable(any), optional
            This variable length list of tensors is used as additional arguments
            to evaluate the network. These additional arguments are passed to
            the forward method of the network.
        **kwargs : dict(str, any), optional
            These additional keyword arguments are used as additional arguments
            to evaluate the network. These additional keyword arguments are
            passed to the forward method of the network.

        Returns
        -------
        gen_loss : py:class:`torch.Tensor`
            The estimated generator loss based on given data and trained
            network. This generator loss has the same tensor type as given
            `fake_data`.
        """
        batch_size = fake_data.size()[0]

        fake_critic = self.forward(fake_data, *args, **kwargs)
        real_labels = self.get_targets(batch_size, 1.0, fake_data)
        gen_loss = self.disc_loss(fake_critic, real_labels)
        return gen_loss

    def recon_loss(self, recon_obs, *args, **kwargs):
        """
        This reconstruction loss is used for the autoencoder to nudge the
        reconstructed observations to the real observations.

        Parameters
        ----------
        recon_obs : :py:torch:`torch.Tensor`
            The reconstruction loss is estimated based on these reconstructed
            observations.
        *args : iterable(any), optional
            This variable length list of tensors is used as additional arguments
            to evaluate the network. These additional arguments are passed to
            the forward method of the network.
        **kwargs : dict(str, any), optional
            These additional keyword arguments are used as additional arguments
            to evaluate the network. These additional keyword arguments are
            passed to the forward method of the network.

        Returns
        -------
        recon_loss : :py:class:`torch.Tensor`
            This reconstruction loss is estimated on given reconstructed
            observations and the trained network. This reconstruction loss has
            the same tensor type as `recon_obs`.

        Notes
        -----
        This method passes `recon_obs` as fake data to `gen_loss`, which
        estimates the reconstruction loss.
        """
        recon_loss = self.gen_loss(recon_obs, *args, **kwargs)
        return recon_loss

    def back_loss(self, analysis, *args, **kwargs):
        """
        This background loss is used for the autoencoder to nudge the analysis
        to the prior.

        Parameters
        ----------
        analysis : :py:torch:`torch.Tensor`
            The background loss is estimated based on this estimated analysis.
        *args : iterable(any), optional
            This variable length list of tensors is used as additional arguments
            to evaluate the network. These additional arguments are passed to
            the forward method of the network.
        **kwargs : dict(str, any), optional
            These additional keyword arguments are used as additional arguments
            to evaluate the network. These additional keyword arguments are
            passed to the forward method of the network.

        Returns
        -------
        back_loss : :py:class:`torch.Tensor`
            This background loss is estimated on given analysis and the trained
            network. This background loss has the same tensor type as
            `recon_obs`.

        Notes
        -----
        This method passes `analysis` as fake data to `gen_loss`, which
        estimates the background loss.
        """
        back_loss = self.gen_loss(analysis, *args, **kwargs)
        return back_loss
