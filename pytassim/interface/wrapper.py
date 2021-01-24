#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 22.01.21
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Callable

# External modules
import torch
import numpy as np

# Internal modules
from pytassim.core.base import BaseModule
from pytassim.localization.localization import BaseLocalization


logger = logging.getLogger(__name__)


def wrapper_bridge(
        core_module: BaseModule,
        device: torch.device,
        dtype: torch.dtype
) -> Callable:
    """
    This bridged module is the module of the algorithm bridged to work
    with numpy array.
    The bridged module function will returned the ensemble weights as numpy
    array with the same dtype as the first argument to the wrapped module.

    Parameters
    ----------
    core_module : child of pytassim.core.base.BaseModule
        This core module is wrapped to use numpy arrays.
    device : torch.device
        The pytorch data will be copied to this device.
    dtype : torch.dtype
        The pytorch data will be transformed into this format.

    Returns
    -------
    wrapped_module : func
        This is the bridged module, which uses as in- and output numpy arrays.
    """
    def bridged_module(*args):
        torch_args = [
            torch.from_numpy(arg).to(device=device, dtype=dtype)
            for arg in args
        ]
        torch_weights = core_module(*torch_args)
        torch_weights = torch_weights.cpu().detach()
        weights = torch_weights.numpy().astype(args[0].dtype)
        return weights
    return bridged_module


def wrapper_localization(
        module: Callable,
        localization: BaseLocalization
) -> Callable:
    """
    This wrapper localizes a given module with given localization.

    Parameters
    ----------
    module : child of pytassim.core.base.BaseModule
        This module will be wrapped by this wrapping function.
    localization : child of pytassim.localization.base.BaseLocalization
        This initialized localization will be used to localize the states
        within the given wrapped module.

    Returns
    -------
    localized_module : func
        This is the localized module.
    """
    def localized_module(grid_info, *args, obs_info=None, args_to_skip=None):
        if localization is not None:
            luse, lweights = localization.localize_obs(
                grid_info, obs_info
            )
            lweights = np.sqrt(lweights[luse])
            if args_to_skip is None:
                args_to_skip = []
            args = [
                arg if k in args_to_skip else arg[..., luse] * lweights
                for k, arg in enumerate(args)
            ]
        return module(*args)
    return localized_module
