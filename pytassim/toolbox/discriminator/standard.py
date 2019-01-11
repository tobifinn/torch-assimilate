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
    loss_func : py:class:`pytorch.nn.Module`
        This loss function should be an initialized loss function from
        :py:mod:`torch.nn`.
    """
    def __init__(self, net,):
        self.net = net
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.optimizer = None

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
