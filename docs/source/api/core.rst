This is the core for all data assimilation algorithms. The core is purely
writte in PyTorch to improve its speed and increase its differentiability.


Modules
-------
These modules are the core of this data assimilation.

.. autoclass:: pytassim.core.etkf.ETKFModule
    :members:
    :undoc-members:
    :show-inheritance:


.. automodule:: pytassim.core.ketkf.KETKFModule
    :members:
    :undoc-members:
    :show-inheritance:


Utilities
---------
These utilities are commonly used across all modules.

.. automodule:: pytassim.core.utils
    :members:
    :undoc-members:
    :show-inheritance:
