.. tf-assimilate documentation master file, created by
   sphinx-quickstart on Sun Mar 11 12:19:28 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tf-assimilate is a package for data assimilation based on tensorflow
====================================================================
tf-assimilate is a python package for data assimilation of meteorological
observations into numerical weather model data.

This package is constructed for efficient and parallelized data assimilation in
python. The central entity of this package are the data assimilation methods
optimized in tensorflow [1]_. For data in- and output xarray [2]_ is used.
Originally, this package can be used for offline data assimilation.

In the future, different data assimilation methods, like
ensemble Kalman filters, particle filters and variational data assimilation will
be added.

This package is developed for a PhD-thesis about nonlinear methods in data
assimilation at the "Universität Hamburg".

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
----------
.. [1] Tensorflow, https://www.tensorflow.org/
.. [2] xarray, http://xarray.pydata.org
