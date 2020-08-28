Kernelized Ensemble Transform Kalman filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The kernelized ensemble transform Kalman filter (KETKF) is a kernelized version
of the ensemble transform Kalman filter (ETKF).
The KETKF estimates ensemble
weights similar to the standard ETKF.
The ETKF relies on a linear mapping
from observations to weight space, whereas the KETKF includes a non-linear
mapping via the kernel trick.

For the KEKTF, the weights are estimated globally for either correlated
:py:class:`pytassim.assimilation.filter.ketkf.KETKFCorr` or uncorrelated
:py:class:`pytassim.assimilation.filter.ketkf.KETKFUncorr` observations.
The implementation further allows filtering in time similar to
:cite:`hunt_four-dimensional_2004` and ensemble smoothing.

A prior regularization factor, called `inflation factor` can be chosen.
This inflation factor modifies the prior covariance of the ensemble weights
such that observations have either more or less influence on the analysis.

.. autosummary::
    pytassim.assimilation.filter.ketkf.KETKFCorr
    pytassim.assimilation.filter.ketkf.KETKFUncorr