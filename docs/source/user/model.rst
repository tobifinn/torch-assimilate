Model interface and integration
===============================



Model interface
---------------
The model interface is designed such that real-world model output can be used
for this data assimilation package. Every supported model has a pre- and
post-process function, which is either used to get data into a valid model state
or a valid model state into model output.

COnsortium for Small-scale MOdeling (COSMO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The COnsortium for Small-scale Modeling model :cite:`baldauf_operational_2011`,
is the operational weather forecast model of the German Weather Service (DWD)
for an area around Germany. It is also used within the coupled model system
TerrSysMP :cite:`shrestha_scale-consistent_2014`.
.. autosummary::
    pytassim.model.terrsysmp.cosmo.preprocess_cosmo
    pytassim.model.terrsysmp.cosmo.post_process_cosmo

Lorenz '96
----------

.. autosummary::
    pytassim.model.lorenz_96.lorenz_96.Lorenz96
    pytassim.model.lorenz_96.forward_model.forward_model


Lorenz '84
----------

.. autosummary::
    pytassim.model.lorenz_84.lorenz_84.Lorenz84


ODE Integration
---------------

Runge-Kutta
^^^^^^^^^^^

.. autosummary::
    pytassim.model.integration.rk4.RK4Integrator
    pytassim.model.integration.integrator.BaseIntegrator