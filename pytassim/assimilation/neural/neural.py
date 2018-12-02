#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 26.03.18
#
# Created for tf-assimilate
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

# External modules

# Internal modules
from ..base import BaseAssimilation


logger = logging.getLogger(__name__)


class NeuralAssimilation(BaseAssimilation):
    """
    NeuralAssimilation is a base class for all data assimilation algorithms,
    which are based on neural networks This class can be used for fast
    prototyping of these algorithms. For fast prototyping some attributes and
    methods are added compared to :py:class:``BaseAssimilation``.
    """
    pass
