This package is used to unify different data assimilation algorithms.


Algorithms
----------
Here are all available and implemented classical data assimilation algorithms
listed.

.. toctree::
   :maxdepth: 1

   algorithms/etkf
   algorithms/letkf
   algorithms/ketkf

Neural assimilation
-------------------
Here is the neural assimilation class and available neural networks shown.

.. automodule:: pytassim.assimilation.neural.neural
    :members:
    :undoc-members:
    :show-inheritance:


Base classes
------------
All data assimilation algorithms are inherited from one of these base classes.

.. automodule:: pytassim.assimilation.base
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: pytassim.assimilation.filter.filter
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: pytassim.assimilation.variational.variational
    :members:
    :undoc-members:
    :show-inheritance:
