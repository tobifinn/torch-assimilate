
# coding: utf-8

# In[1]:


import xarray as xr
import pandas as pd
import numpy as np

import pytassim.state
import pytassim.observation


# In[2]:


rnd = np.random.RandomState(42)

NR_ENS_MEMS = 10
NR_VARS = 2
NR_TIMES = 3
NR_GRID_POINTS = 40

OBS_ERR = 0.5
OBS_CORR = 0.5


# In[3]:


state_arr = rnd.normal(size=(NR_VARS, NR_TIMES, NR_ENS_MEMS, NR_GRID_POINTS))
ens_mems = np.arange(NR_ENS_MEMS)
var_names = ['x', 'y']
times = pd.date_range(start='1992-12-25', periods=NR_TIMES, freq='H')
grid_points = np.arange(NR_GRID_POINTS)

state = xr.DataArray(
    data = state_arr,
    coords = {
        'var_name': var_names,
        'time': times,
        'ensemble': ens_mems,
        'grid': grid_points
    },
    dims = ('var_name', 'time', 'ensemble', 'grid')
)

state.to_netcdf('test_state.nc')


# In[4]:


print('Ensemble state is valid: {0}'.format(state.state.valid))


# In[5]:


raw_obs = state.mean('ensemble').sel(var_name='x')
raw_cov = xr.DataArray(
    data = np.zeros((raw_obs.grid.size, raw_obs.grid.size)),
    coords = {
        'obs_grid_1': (('obs_grid_1', ), raw_obs.grid.values,),
        'obs_grib_2': (('obs_grid_2', ), raw_obs.grid.values,)
    },
    dims = ['obs_grid_1', 'obs_grid_2']
)


# In[6]:


single_obs = raw_obs + rnd.normal(scale=OBS_ERR, size=raw_obs.shape)
single_obs = single_obs.rename(grid='obs_grid_1')
single_cov_data = np.identity(single_obs.obs_grid_1.size) * OBS_ERR
single_cov = raw_cov.copy(deep=True, data=single_cov_data)


# In[7]:


single_ds = xr.Dataset(
    {
        'observations': single_obs,
        'covariance': single_cov
    }
)
single_ds.to_netcdf('test_single_obs.nc')


# In[8]:


print('Single observations are valid: {0}'.format(single_ds.obs.valid))

