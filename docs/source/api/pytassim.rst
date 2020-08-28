torch-assimilate package
========================

Assimilation algorithms
-----------------------
.. toctree::
   :maxdepth: 1

   assimilation

Localization
------------
.. toctree::
   :maxdepth: 1

   localization

Kernels
-------
.. toctree::
   :maxdepth: 1

   kernels

Models
------
.. toctree::
   :maxdepth: 1

   model

Observation operators
---------------------
.. toctree::
   :maxdepth: 1

   obs_ops

Testing
-------
.. toctree::
   :maxdepth: 1

   testing

Observation subset
------------------
A :py:class:`xarray.Dataset` accessor, accessible with
:py:attr:`xarray.Dataset.obs`.

.. automodule:: pytassim.observation
    :members:
    :undoc-members:
    :show-inheritance:

Model state
-----------
A :py:class:`xarray.DataArray` accessor, accessible with
:py:attr:`xarray.DataArray.state`.

.. automodule:: pytassim.state
    :members:
    :undoc-members:
    :show-inheritance:
