#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 18.12.20
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


def lazy_property(attr_name=None):
    def lazy_wrap(fn):
        """
        Decorator that makes a property lazy-evaluated.
        Based on: http://stevenloria.com/lazy-evaluated-properties-in-python/
        """
        if isinstance(attr_name, str):
            prvt_name = '_{0:s}'.format(attr_name)
        elif attr_name is None:
            prvt_name = '_{0:s}'.format(fn.__name__)
        else:
            raise TypeError(
                'Given attribute name has to be a str or None, instead it has '
                '{0} as type'.format(type(attr_name))
            )

        @property
        def _lazy_property(self):
            if not hasattr(self, prvt_name) or getattr(self, prvt_name) is None:
                setattr(self, prvt_name, fn(self))
            return getattr(self, prvt_name)
        return _lazy_property
    return lazy_wrap


def ensure_tensor(func):
    def wrapped_func(self, new_value):
        if not isinstance(new_value, (torch.Tensor, torch.nn.Parameter)):
            new_value = torch.tensor(new_value, dtype=self.dtype,
                                     device=self.device)
        return func(self, new_value)
    return wrapped_func


def bound_tensor(min_val=None, max_val=None):
    def bound_wrap(func):
        def bound_func(self, new_value):
            if (min_val is not None) and torch.any(new_value < min_val):
                raise ValueError(
                    'Given new value {0} is smaller than the minimum value '
                    '{1}!'.format(new_value, min_val)
                )
            if (max_val is not None) and torch.any(new_value > max_val):
                raise ValueError(
                    'Given new value {0} is larger than the maximum value '
                    '{1}!'.format(new_value, max_val)
                )
            return func(self, new_value)
        return bound_func
    return bound_wrap
