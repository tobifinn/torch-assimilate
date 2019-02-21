This submodule includes different models and integration solutions, and can be
used to test the efficiency of different data assimilation algorithms.

Model interface
---------------
This model interface coupled real-world model output with this data assimilation
framework.

.. automodule:: pytassim.model.terrsysmp.cosmo
    :members:
    :undoc-members:
    :show-inheritance:

Models
------
This documents all available and tested models. These models have to be
initialized and can be afterward called, like functions.

.. automodule:: pytassim.model.lorenz_84
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: pytassim.model.lorenz_96
    :members:
    :undoc-members:
    :show-inheritance:

Integration
-----------
These integration algorithms are used to integrate given models forward or
backward in time. Every integration algorithm should inherit from
``BaseIntegrator``.

.. automodule:: pytassim.model.integration.integrator
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: pytassim.model.integration.rk4
    :members:
    :undoc-members:
    :show-inheritance:
