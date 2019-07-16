#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 7/16/19
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
import unittest
import os

# External modules
import xarray as xr
from dask.distributed import LocalCluster, Client

# Internal modules
from pytassim.testing import dummy_obs_operator

logger = logging.getLogger(__name__)


class TestDistributedCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cluster = LocalCluster(
            n_workers=1, threads_per_worker=1, local_dir="/tmp/dask_work",
            processes=False
        )
        cls.client = Client(cls.cluster)
        cls.state = xr.open_dataarray(cls.state_path).load()
        cls.obs = xr.open_dataset(cls.obs_path).load()
        cls.obs.obs.operator = dummy_obs_operator

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        cls.cluster.close()
