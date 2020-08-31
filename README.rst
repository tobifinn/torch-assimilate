torch-assimilate
================

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
.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4005994.svg
    :target: https://doi.org/10.5281/zenodo.4005994



.. end_badges

Data assimilation based on PyTorch
-------------------------------------

torch-assimilate is a python package for data assimilation of meteorological
observations into numerical weather model data.

This package is constructed for efficient and parallelized data assimilation in
python. The central entity of this package are the data assimilation methods
optimized in PyTorch [1]_. Furthermore, some
algorithms are parallelized with dask [2]_ and allow a distributed computing
with many cores. For data in-  and output xarray [3]_ is
used. Originally, this package is designed for offline data assimilation via
io-operations.

In the future, different data assimilation methods, like ensemble Kalman
filters, particle filters, variational data assimilation and neural assimilation
will be added.

This package is developed for a PhD-thesis about nonlinear methods in
coupled data assimilation at the "Universität Hamburg", "Universität Bonn"
and the Max Planck Institute for Meteorology.

Installation
------------
We highly recommend to create a virtual environment for this package to prevent
package collisions.
At the moment this package is only available at pypi-test.

This package is programmed in python 3.6 and should be working with all `python
versions > 3.3`. Additional requirements are pytorch and xarray.

PyTorch needs to be additionally installed because of different possible versions. In following CPU-based installation for linux is shown.

via conda (recommended):
^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: sh

    git clone git@gitlab.com:tobifinn/torch-assimilate.git
    cd torch-assimilate
    conda env create -f environment.yml
    source activate pytassim
    conda install pytorch torchvision cpuonly -c pytorch
    pip install .

via pip (latest pypi-test):
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: sh

    pip install --index-url https://test.pypi.org/simple/ torch-assimilate
    pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

Authors
-------
* **Tobias Finn** - *Initial creator* - `tobifinn <gitlab.com/tobifinn>`_

License
-------

This project is licensed under the GPL3 License - see the
`license <LICENSE.md>`_ file for details.

References
----------
.. [1] PyTorch, https://pytorch.org
.. [2] Dask, https://dask.org
.. [3] xarray, http://xarray.pydata.org
