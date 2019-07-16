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
from unittest.mock import MagicMock
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

    def test_cluster_gets_private_cluster(self):
        self.algorithm._cluster = None
        self.assertIsNone(self.algorithm.cluster)
        self.algorithm._cluster = 1234
        self.assertEqual(self.algorithm.cluster, 1234)

    def test_client_gets_private_client(self):
        self.algorithm._client = None
        self.assertIsNone(self.algorithm.client)
        self.algorithm._client = 12345
        self.assertEqual(self.algorithm.client, 12345)


    def test_validate_client_checks_if_client_is_client(self):
        self.assertTrue(self.algorithm._validate_client(self.client))
        self.assertFalse(self.algorithm._validate_client(1234))

    def test_validate_cluster_checks_if_cluster_has_scheduler_address(self):
        self.assertTrue(self.algorithm._validate_cluster(self.cluster))
        self.assertFalse(self.algorithm._validate_cluster(12345))

    def test_check_client_cluster_checks_if_client_or_cluster_given(self):
        self.algorithm._check_client_cluster(self.client, self.cluster)
        self.algorithm._check_client_cluster(self.client, None)
        self.algorithm._check_client_cluster(None, self.cluster)
        with self.assertRaises(ValueError):
            self.algorithm._check_client_cluster(None, None)

    def test_set_client_cluster_calls_check_client_cluster(self):
        self.algorithm._check_client_cluster = MagicMock()
        self.algorithm.set_client_cluster(self.client, self.cluster)
        self.algorithm._check_client_cluster.assert_called_once_with(
            self.client, self.cluster
        )

    def test_set_client_cluster_sets_client_if_given(self):
        self.algorithm._client = None
        self.algorithm.set_client_cluster(self.client, None)
        self.assertEqual(id(self.algorithm._client), id(self.client))

    def test_set_client_cluster_sets_cluster_from_client(self):
        self.algorithm._cluster = None
        self.algorithm.set_client_cluster(self.client, None)
        self.assertEqual(id(self.algorithm._cluster), id(self.client.cluster))

    def test_set_client_cluster_sets_cluster_from_cluster_if_no_client(self):
        self.algorithm._cluster = None
        self.algorithm.set_client_cluster(None, self.cluster)
        self.assertEqual(id(self.algorithm._cluster), id(self.cluster))

    def test_set_client_cluster_initialize_client_from_cluster(self):
        self.algorithm._client = None
        self.algorithm.set_client_cluster(None, self.cluster)
        self.assertIsInstance(self.algorithm._client, Client)
        self.assertEqual(id(self.algorithm._client.cluster), id(self.cluster))
        self.algorithm.client.close()

    def test_set_client_cluster_uses_client_if_both_given(self):
        self.algorithm._client = None
        self.algorithm.set_client_cluster(self.client, self.cluster)
        self.assertEqual(id(self.algorithm._client), id(self.client))