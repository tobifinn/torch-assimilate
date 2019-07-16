import unittest
import os

import xarray as xr
from dask.distributed import LocalCluster, Client

from pytassim.testing import dummy_obs_operator


BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestDistributedAssimilation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cluster = LocalCluster(
            n_workers=1, threads_per_worker=1, local_dir="/tmp/dask_work",
            processes=False
        )
        cls.client = Client(cls.cluster)
        state_path = os.path.join(DATA_PATH, 'test_state.nc')
        cls.state = xr.open_dataarray(state_path).load()
        obs_path = os.path.join(DATA_PATH, 'test_single_obs.nc')
        cls.obs = xr.open_dataset(obs_path).load()
        cls.obs.obs.operator = dummy_obs_operator

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        cls.cluster.close()
