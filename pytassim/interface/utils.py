#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 03.01.21
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


def bridge_decorator(module, device, dtype):
    """
    This decorator is a bridge from the Xarray/Dask-interface to the core
    modules.
    The wrapped module function will returned the ensemble weights as numpy
    array with the same dtype as the first argument to the wrapped module.

    Parameters
    ----------
    module : torch.nn.Module
        This module will be wrapped by this decorator and should have the
        ensemble weights as only return parameter.
    device : torch.device
        The arguments for the wrapped module are transferred to this device.
    dtype : torch.dtype
        The arguments for the wrapped module are casted into this dtype.

    Returns
    -------
    wrapped_module : func
        This is the wrapped module function.
    """
    def wrapped_module(*args):
        torch_args = [
            torch.from_numpy(arg).to(device=device, dtype=dtype)
            for arg in args
        ]
        torch_weights = module(*torch_args)
        torch_weights = torch_weights.cpu().detach()
        weights = torch_weights.numpy().astype(args[0].dtype)
        return weights
    return wrapped_module
