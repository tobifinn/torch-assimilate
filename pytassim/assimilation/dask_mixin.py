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

# External modules
from distributed import Client

# Internal modules


logger = logging.getLogger(__name__)


class DaskMixin(object):
    """
    This mixin can be used to define a common dask interface. This mixin defines
    method for specifying dask operators and helps to set-up a client.
    """
    def __init__(self, client=None, cluster=None, chunksize=10, **kwargs):
        super().__init__(**kwargs)
        self._cluster = None
        self._client = None
        self._chunksize = 1
        self.chunksize = chunksize
        self.set_client_cluster(client=client, cluster=cluster)

    @staticmethod
    def _validate_client(client):
        return isinstance(client, Client)

    @staticmethod
    def _validate_cluster(cluster):
        return hasattr(cluster, "scheduler_address")

    def _check_client_cluster(self, client, cluster):
        not_valid_cluster = not self._validate_cluster(cluster)
        not_valid_client = not self._validate_client(client)
        if not_valid_client and not_valid_cluster:
            raise ValueError(
                'Either a client or a cluster have to be specified!'
            )

    @property
    def cluster(self):
        return self._cluster

    @property
    def client(self):
        return self._client

    def set_client_cluster(self, client=None, cluster=None):
        """
        This method sets the client and cluster. If both are given and
        valid,
        then client has priority.

        Parameters
        ----------
        client : :py:class:``~dask.distributed.Client`` or None
            This dask distributed client is used to parallelize the
            processes.
            Either this client or ``cluster`` has to be
            specified. Default is None.
        cluster : compatible to
        :py:class:``~dask.disributed.LocalCluster`` or None
            This dask cluster is used to initialize a
            :py:class:``~dask.distributed.Client``, if no client is
            specified.
            Default is None.
        """
        self._check_client_cluster(client, cluster)
        if self._validate_client(client):
            self._client = client
            self._cluster = client.cluster
        else:
            self._cluster = cluster
            self._client = Client(cluster)

