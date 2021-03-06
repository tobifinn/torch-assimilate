Ensemble Transform Kalman filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ensemble transform Kalman filter (ETKF) :cite:`bishop_adaptive_2001` is an
ensemble Kalman filter, where
the analysis is generated based on estimated weights. These weights are
generated by using the Kalman filter equations in a reduced weight space. The
ETKF is further a square-root ensemble Kalman filter, whre the ensemble
perturbations are estimated deterministically. The computing time is largely
determined by the observational size and the speed of the observation operators.

For the EKTF, the weights are estimated globally with
:py:class:`pytassim.core.etkf.ETKFModule` as core module. The
implementation follows the equations of :cite:`hunt_efficient_2007`. The
implementation further allows filtering in time based on linear propagation
assumption :cite:`hunt_four-dimensional_2004` and ensemble smoothing.

As forgetting factor in time, an inflation factor can be chosen. This inflation
factor is used to artificially inflate the background weights and leads to an
inflated analysis ensemble.

There will be also implementations of the ETKF for both interfaces – the
Xarray/Dask interface and the Light interface.

.. autosummary::
    pytassim.interface.etkf.ETKF
    pytassim.core.etkf.ETKFModule

Localized Ensemble Transform Kalman filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The localized ensemble transform Kalman filter (LETKF)
:cite:`hunt_efficient_2007` is a localized implementation of the ETKF. In this
localized implementation the ensemble weights are independently estimated for
every grid point. Thus, the computing time is now dependent on the observational
size and the number of grid points within the ensemble model state.

Caused by the domain decomposition, the most important argument for the LETKF is
the chosen localization. Here, it is possible to choose any localization
specified within `pytassim.localization`, which supports observational
localization. If no localization is used, the analysis will be the same as for
the ETKF, but only estimated in an inefficient way.

As for the ETKF, the weights are estimated based on the
:py:class:`pytassim.core.etkf.ETKFModule` core module. The
implementation further allows filtering in time based on linear propagation
assumption :cite:`hunt_four-dimensional_2004` and ensemble smoothing.

As forgetting factor in time, an inflation factor can be chosen. This inflation
factor is used to artifically inflate the background weights and leads to an
inflated analysis ensemble.

The Xarray/Dask interface for the LETKF iterates over the state time and grid
and estimates for every temporal and spatial point independent weights. The
implementation uses :py:func:`xarray.apply_ufunc` and is able to fully
exploit a distributed environment with ``dask.distributed``. For efficiency
different `chunksizes` can be specified. To increase the speed of the
algorithm it is recommended to set ``OMP_NUM_THREADS=1`` as environment
variable, until some pytorch functions are parallelized.

.. autosummary::
    pytassim.interface.letkf.LETKF
