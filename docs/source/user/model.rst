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

An opened NetCDF4 COSMO file can be
converted into a valid model state with
:py:func:`~pytassim.model.terrsysmp.cosmo.preprocess_cosmo`. The resulting model
state has an unified grid based on an Arakawa-A
:cite:`arakawa_computational_1977` grid and on half levels. All variables are
vertically remapped via nearest neighbor to these half levels. Variables without
vertical coordinates are assigned to the lowest half level. It is possible to
select variables, which should be used for data assimilation.

An analysis array can be converted into a valid COSMO dataset by using
:py:func:`~pytassim.model.terrsysmp.cosmo.postprocess_cosmo`. This function
returns a given COSMO dataset, where analysed variables replace their forecasted
counterpart.

.. autosummary::
    pytassim.model.terrsysmp.cosmo.preprocess_cosmo
    pytassim.model.terrsysmp.cosmo.postprocess_cosmo

Community Land Model (CLM)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Community Land Model (CLM) is a land surface model written by the National
Center for Atmospheric Research (NCAR)
:cite:`oleson_technical_2004,oleson_k._w._improvements_2008`. CLM is the soil
and land model for the Community Earth System Model and TerrSysMP.

An opened NetCDF4 CLM file can be converted into a valid model state with
:py:func:`~pytassim.model.terrsysmp.cln.preprocess_clm`. The resulting model
state has an unified vertical grid. All vertical grids of variables are remapped
to this unified vertical grid. Variables without vertical coordinates are
assigned to the highest depth. It is also possible to select variables, which
are used for data assimilation.

An analysis array can be converted into a valid CLM dataset by using
:py:func:`~pytassim.model.terrsysmp.clm.postprocess_clm`. This function returns
given and modified CLM dataset, where analysed variables replace their
forecasted counterpart.

.. autosummary::
    pytassim.model.terrsysmp.clm.preprocess_clm
    pytassim.model.terrsysmp.clm.postprocess_clm


Lorenz '96
----------

.. autosummary::
    pytassim.model.lorenz_96.Lorenz96


Lorenz '84
----------

.. autosummary::
    pytassim.model.lorenz_84.Lorenz84


ODE Integration
---------------

Runge-Kutta
^^^^^^^^^^^

.. autosummary::
    pytassim.model.integration.rk4.RK4Integrator
    pytassim.model.integration.integrator.BaseIntegrator