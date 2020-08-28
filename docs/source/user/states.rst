Model states and observations
=============================
Model states and observations are used as input and output to any assimilation
algorithm. They are based on :py:class:`xarray.DataArray` and
:py:class:`xarray.Dataset` with specific accessors for either model states or
observations. The observation covariance / uncertainty and the observation
operators are also encapsulated within the observational
:py:class:`xarray.Dataset` and its accessor. The accessors are further used to
check if given fields are valid for the assimilation.

Model states
------------
The model states are represented as :py:class:`xarray.DataArray` with
coordinates, which are explained in the following:

    ``var_name`` – str
        This coordinate allows to concatenate multiple variables into
        one single array. The coordinate should have :py:class:`str` as
        dtype.

    ``time`` – datetime like
        Different times can be concatenated into this coordinate. This
        coordinate is useful for data assimilation algorithms which
        allow assimilation of time dependent variables. The coordinate
        should be a datetime like dtype.

    ``ensemble`` – int
        This coordinate is used for ensemble based data assimilation
        algorithms. The ensemble members are numbered as
        :py:class:`int`. An integer value of 0 symbolizes a
        deterministic or control run. In some ensemble based algorithms,
        the ensemble coordinate is looped.

    ``grid``
        This coordinate is used to characterize the spatial position
        of one single value within the array. This coordinate can be a
        :py:class:`~pandas.MultiIndex`, where different coordinate
        components are stacked. In some algorithms the analysis is
        calculated for every grid point independently such that the
        algorithm loops over this coordinate.


Observations
------------
Observations are represented as observation subsets. Within an observation
subset, the observations can be spatial correlated. Different observation
subsets are per definition uncorrelated. Observation subsets are
:py:class:`~xarray.Dataset`s with two variables:

        observations
            (time, obs_grid_1), the actual observation values

        covariance
            (obs_grid_1) or (obs_grid_1, obs_grid_2), the covariance between
            different observations. If this is a vector, then it is assumed
            that the observations are uncorrelated and only the variances
            are within this array.

An additional accessor is registered under :py:attr:`xarray.Dataset.obs`, where
the here additional attributes and methods can be accessed. It is possible to
check the validity of given observation subsets by calling
`xarray.Dataset.obs.valid`. Further it is possible to see if the observations
are correlated or not with `xarray.Dataset.obs.correlated`. The assimilation
algorithms will use the abstract method :py:meth:`xarray.Dataset.obs.operator`
to convert the given model field into an observation equivalent. Thus, the
`operator` method needs to be overwritten for every observation subset
independently.


References
----------
.. autosummary::
    pytassim.state.ModelState
    pytassim.observation.Observation
    xarray.DataArray
    xarray.Dataset