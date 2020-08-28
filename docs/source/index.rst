PyTorch based data assimilation package
=======================================

.. start_badges

.. list-table::
    :stub-columns: 1
    :widths: 15 85

    * - docs
      - |docs|
    * - pipeline
      - |pipeline|
    * - coverage
      - |coverage|
    * - quality
      - |quality|
    * - package
      - |pypi-test| |pypi| |conda| |zenodo|

.. |pipeline| image:: https://gitlab.com/tobifinn/torch-assimilate/badges/dev/pipeline.svg
    :target: https://gitlab.com/tobifinn/torch-assimilate/pipelines
    :alt: Pipeline status
.. |coverage| image:: https://gitlab.com/tobifinn/torch-assimilate/badges/dev/coverage.svg
    :target: https://tobifinn.gitlab.io/torch-assimilate/coverage-report/
    :alt: Coverage report
.. |docs| image:: https://img.shields.io/badge/docs-sphinx-brightgreen.svg
    :target: https://tobifinn.gitlab.io/torch-assimilate/
    :alt: Documentation Status
.. |quality| image:: https://img.shields.io/badge/quality-codeclimate-brightgreen.svg
    :target: https://tobifinn.gitlab.io/torch-assimilate/coverage-report/codeclimate.html
.. |pypi| image:: https://img.shields.io/badge/pypi-unavailable-lightgrey.svg
    :target: https://pypi.org/project/torch-assimilate/
.. |pypi-test| image:: https://img.shields.io/badge/pypi_test-online-brightgreen.svg
    :target: https://test.pypi.org/project/torch-assimilate/
.. |conda| image:: https://img.shields.io/badge/conda-unavailable-lightgrey.svg
    :target: https://anaconda.org/tobifinn/torch-assimilate
.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4005995.svg
    :target: https://doi.org/10.5281/zenodo.4005995

.. end_badges

torch-assimilate is a python package for data assimilation of meteorological
observations into numerical weather model data.

This package is constructed for efficient and parallelized data assimilation in
python. The central entity of this package are the data assimilation methods
optimized in PyTorch :cite:`paszke_automatic_2017`. For data in- and output
xarray :cite:`hoyer_xarray_2017` is used.
Originally. this package can be used for offline data assimilation.

In the future, different data assimilation methods, like
ensemble Kalman filters, particle filters, variational data assimilation and neural assimilation will
be added.

This package is developed for a PhD-thesis about nonlinear methods in data
assimilation at the "Universität Hamburg".

Documentation
-------------
.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   start/overview
   start/installation

.. toctree::
   :maxdepth: 2
   :caption: Scientific User Guide:

   user/states
   user/assimilation
   user/localization
   user/transform
   user/obs_ops
   user/model

.. toctree::
   :maxdepth: 1
   :caption: Help & References:

   api/pytassim
   appendix/references
   appendix/publications
