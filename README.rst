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
      - |pypi-test| |pypi| |conda|

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

.. end_badges

Data assimilation based on PyTorch
-------------------------------------

torch-assimilate is a python package for data assimilation of meteorological
observations into numerical weather model data.

This package is constructed for efficient and parallelized data assimilation in
python. The central entity of this package are the data assimilation methods
optimized in PyTorch [1]_. For data in- and output xarray [2]_ is used.
Originally. this package can be used for offline data assimilation.

In the future, different data assimilation methods, like
ensemble Kalman filters, particle filters, variational data assimilation and neural assimilation will
be added.

This package is developed for a PhD-thesis about nonlinear methods in data
assimilation at the "Universität Hamburg".

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
    conda install pytorch-cpu torchvision-cpu -c pytorch
    pip install .

via pip (latest pypi-test):
^^^^^^^^
.. code:: sh

    pip install --index-url https://test.pypi.org/simple/ torch-assimilate
    pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
    pip3 install torchvision

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
.. [2] xarray, http://xarray.pydata.org
