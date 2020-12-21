Kernelized Ensemble Transform Kalman filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The kernelized ensemble transform Kalman filter (KETKF) is a kernelized version
of the ensemble transform Kalman filter (ETKF).
The KETKF estimates ensemble
weights similar to the standard ETKF.
The ETKF relies on a linear mapping
from observations to weight space, whereas the KETKF includes a non-linear
mapping via the kernel trick.

For the KEKTF, the weights are estimated globally with
:py:class:`pytassim.core.ketkf.KETKFModule` as core module.
The implementation further allows filtering in time similar to
:cite:`hunt_four-dimensional_2004` and ensemble smoothing.

A prior regularization factor, called `inflation factor` can be chosen.
This inflation factor modifies the prior covariance of the ensemble weights
such that observations have either more or less influence on the analysis.

There will be also implementations of the KETKF for both interfaces – the
Xarray/Dask interface and the Light interface.

.. autosummary::
    pytassim.interface.ketkf.KETKF
    pytassim.core.ketkf.KETKFModule


Localized and Kernelized Ensemble Transform Kalman filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Similar to the KETKF above, the localized and kernelized ensemble transform
Kalman filter (LKETKF) is a non-linear modification of the LETKF. This
modification uses the kernel trick to allow non-linear mappings from
observational space into ensemble space.In this localized implementation the
ensemble weights are independently estimated for every grid point. Thus, the
computing time is now dependent on the observational size and the number of
grid points within the ensemble model state.

Caused by the domain decomposition, the most important argument for the
LKETKF is
the chosen localization. Here, it is possible to choose any localization
specified within `pytassim.localization`, which supports observational
localization. If no localization is used, the analysis will be the same as for
the KETKF, but only estimated in an inefficient way.

As for the KETKF, the weights are estimated based on the
:py:class:`pytassim.core.ketkf.KETKFModule` core module. The
implementation further allows filtering in time and ensemble smoothing.

As forgetting factor in time, an inflation factor can be chosen. This inflation
factor is used to artifically inflate the background weights and leads to an
inflated analysis ensemble.

The Xarray/Dask interface for the LKETKF iterates over the state time and grid
and estimates for every temporal and spatial point independent weights. The
implementation uses :py:func:`xarray.apply_ufunc` and is able to fully
exploit a distributed environment with ``dask.distributed``. For efficiency
different `chunksizes` can be specified. To increase the speed of the
algorithm it is recommended to set ``OMP_NUM_THREADS=1`` as environment
variable, until some pytorch functions are parallelized.

.. autosummary::
    pytassim.interface.lketkf.LKETKF
