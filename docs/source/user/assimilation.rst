Assimilation
============
Here we will describe the scientific background behind the data assimilation
algorithms. For many algorithms, there are two version with different suffixes:
corr and uncorr. These are different implementations of the same algorithm for
either correlated or uncorrelated observations. They are split up due to speed
considerations. Often the uncorrelated implementation is faster than the
correlated one.

There are three different types of data assimilation algorithms within this
package. The first type are the filter based algorithms. They are typical
independent from the used dynamical model. Here, we will implement
Kalman filters and particle filters. The second type are the variational data
assimilation algorithms. Normally, they need an adjoint of the observation
operator or/and the dynamical model. Here, we will implement 3D-Var and 4D-Var.
The third type of data assimilation algorithms is new neural network based type.
The main purposed of this third type is to test and use deep learning for data
assimilation.

All assimilation algorithms have a common object-oriented interface, which is
specified within :py:class:`pytassim.assimilation.base.BaseAssimilation`.
Observations can be assimilated into a given model state by using
:py:meth:`pytassim.assimilation.base.BaseAssimilation.assimilate`. This method
prepares the model state and observations with a commonly defined methods and
then calls algorithm-specific methods to update the state. For every algorithm
is is further possible to choose a smoother type of algorithm. If no smoother is
used the observations and model state are localized in time to either a given
analysis time or the latest model state time. This package is based on PyTorch.
If PyTorch is installed with a GPU-backend, then also GPU-based assimilation is
natively supported. To switch GPU-computing on and attribute within every
algorithm can be set.

To pre- and post-process the background or analysis, it is possible to specify
pre- and post-transformers. These transformers manipulate the analysis and/or
background and/or observations. They can be used for covariance inflation,
random perturbations, artificial ensemble generation or normalization.

To assimilate observations into model states an observation operator has to be
defined. Every assimilated observation subset should have its own observation
operator. An observation subset without an observation operator will be dropped
for filter-type and variational-type algorithms. In neural network based data
assimilation, the observation operator is often defined within the network and
is therefore not used.


Filter based data assimilation
------------------------------
In filter based data assimilation, the data assimilation is split into two
parts:

1. State propagation

    In state propagation, the analysis is propagate forward in time with a
    dynamical model. This state propagation is independent from data
    assimilation filter.

2. State update

    The forecasted state, called background, is updated with given observations.
    This updated state is also called analysis and can be used to create a new
    forecast. Here the filter algorithm is used to update the state.

Here, we will implement different types of data assimilation filters. At the
moment only the ensemble transform Kalman filter and the localized ensemble
transform Kalman filter are implemented.

.. toctree::
   :maxdepth: 1

   algorithms/etkf
   algorithms/ketkf

.. autosummary::
    pytassim.assimilation.filter.filter.FilterAssimilation


Variational data assimilation
-----------------------------

Implementations
^^^^^^^^^^^^^^^
.. autosummary::
    pytassim.assimilation.variational.variational.VarAssimilation


Neural data assimilation
------------------------

Implementations
^^^^^^^^^^^^^^^
.. autosummary::
    pytassim.assimilation.neural.neural.NeuralAssimilation


API assimilation
----------------
.. autosummary::
    pytassim.assimilation.base.BaseAssimilation
