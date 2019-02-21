This submodules includes observation operator for different types of models.

Lorenz '96
----------
Here are observation operators for the simplified Lorenz '96 model.

.. automodule:: pytassim.obs_ops.lorenz_96.identity
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: pytassim.obs_ops.lorenz_96.bernoulli
    :members:
    :undoc-members:
    :show-inheritance:

TerrSysMP
---------
Here are observation operators for the coupled model system TerrSysMP, which
couples COSMO, CLM and ParFlow.

.. automodule:: pytassim.obs_ops.terrsysmp.cos_t2m
    :members:
    :undoc-members:
    :show-inheritance:


Base class
----------
All observation operators are inherited from this class with a common interface

.. automodule:: pytassim.obs_ops.base_ops
    :members:
    :undoc-members:
    :show-inheritance:

