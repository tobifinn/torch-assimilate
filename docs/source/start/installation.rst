How to install
==============

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
