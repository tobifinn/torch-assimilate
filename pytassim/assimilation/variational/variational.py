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

# External modules

# Internal modules
from ..base import BaseAssimilation


logger = logging.getLogger(__name__)


class VarAssimilation(BaseAssimilation):
    """
    VarAssimilation is a base class for assimilation with a variational
    approach,like ``3DVar`` or ``4DVar``. These algorithms are fitted to
    trajectories of a weather model (especially 4DVar) such that they need
    different algorithms than the filtering approach. This base class extends
    :py:class:`~pytassim.assimilation.base.BaseAssimilation` with different
    methods and attributes to simplify variational data assimilation
    prototyping.
    """
    def __str__(self) -> str:
        return 'Variational'

    def __repr__(self) -> str:
        return 'Variational'
