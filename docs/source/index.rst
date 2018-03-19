Tensorflow based data assimilation package
==========================================

.. start_badges

.. list-table::
    :stub-columns: 1
    :widths: 15 85

    * - pipeline
      - |pipeline|
    * - coverage
      - |coverage|
    * - docs
      - |docs|

.. |pipeline| image:: https://gitlab.com/tobifinn/tf-assimilate/badges/dev/pipeline.svg
    :target: https://gitlab.com/tobifinn/tf-assimilate/commits/dev
    :alt: Pipeline status
.. |coverage| image:: https://gitlab.com/tobifinn/tf-assimilate/badges/dev/coverage.svg
    :target: https://gitlab.com/tobifinn/tf-assimilate/commits/dev
    :alt: Coverage report
.. |docs| image:: https://img.shields.io/badge/docs-sphinx-brightgreen.svg
    :target: https://tobifinn.gitlab.io/tf-assimilate/
    :alt: Documentation Status
.. end_badges

tf-assimilate is a python package for data assimilation of meteorological
observations into numerical weather model data.

This package is constructed for efficient and parallelized data assimilation in
python. The central entity of this package are the data assimilation methods
optimized in tensorflow [1]_. For data in- and output xarray [2]_ is used.
Originally. this package can be used for offline data assimilation.

In the future, different data assimilation methods, like
ensemble Kalman filters, particle filters and variational data assimilation will
be added.

This package is developed for a PhD-thesis about nonlinear methods in data
assimilation at the "Universität Hamburg".

Documentation
-------------
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   content/structure
   api/tfassim


References
----------
.. [1] Tensorflow, https://www.tensorflow.org/
.. [2] xarray, http://xarray.pydata.org
