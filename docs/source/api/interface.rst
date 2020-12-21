This interface links the core algorithms, written in PyTorch, to the outer
world, efficiently implemented with dask/xarray. There are different variants
of the algorithms, which will be explained in their submodule page.

Algorithms
----------
Here are all available and implemented data assimilation algorithms
listed.

.. toctree::
   :maxdepth: 1

   algorithms/etkf
   algorithms/letkf
   algorithms/ketkf

Base classes
------------
All data assimilation algorithms are inherited from one of these base classes.

.. autoclass:: pytassim.interface.filter.FilterAssimilation
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pytassim.interface.variational.VarAssimilation
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pytassim.interface.base.BaseAssimilation
    :members:
    :undoc-members:
    :show-inheritance:


Mixin classes
-------------
These mixin classes are used to extend the functionality of the algorithms

.. automodule:: pytassim.interface.mixin_local
    :members:
    :undoc-members:
    :show-inheritance:
