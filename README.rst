tf-assimilate
=============


.. list-table::
    :stub-columns: 1
    :widths: 15 85

data assimilation based on tensorflow
-------------------------------------

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

Installation
------------
We highly recommend to create a virtual environment for this package to prevent
package collisions.
At the moment, this package is not uploaded via pypi or conda and this package
needs to be cloned and installed manually.

This package is programmed in python 3.6 and should be working with all `python
versions > 3.3`. Additional requirements are tensorflow [1]_ and xarray [2]_.

via conda (recommended):
^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: sh

    git clone git@gitlab.com:tobifinn/tf-assimilate.git
    cd tf-assimilate
    conda env create -f environment.yml
    source activate tf-assimilate
    pip install .

via pip:
^^^^^^^^
.. code:: sh

    git clone git@gitlab.com:tobifinn/tf-assimilate.git
    cd tf-assimilate
    pip install -r requirements.txt
    pip install .

Authors
-------
* **Tobias Finn** - *Initial creator* -
`tobifinn <gitlab.com/tobifinn>`_

License
-------

This project is licensed under the GPL3 License - see the
`license <LICENSE.md>`_ file for details.

References
----------
.. [1] Tensorflow, https://www.tensorflow.org/
.. [2] xarray, http://xarray.pydata.org
