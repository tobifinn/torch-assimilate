#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# System modules
from setuptools import setup, find_packages
from codecs import open
from os import path
import glob

# External modules

# Internal modules
from tfassim import __version__

BASE = path.abspath(path.dirname(__file__))

try:
    import pypandoc
    long_description = pypandoc.convert(path.join(BASE, 'README.md'), 'rst')
except(IOError, ImportError):
    long_description = open(
        path.join(BASE, 'README.md'), encoding='utf-8').read()


setup(
    name='tf-assimilate',

    version=__version__,

    description='tf-assimilate is a data assimilation package based on '
                'tensorflow',
    long_description=long_description,

    url='https://gitlab.com/tobifinn/tf-assimilate',

    author='Tobias Sebastian Finn',
    author_email='tobias.sebastian.finn@uni-hamburg.de',

    license='GPL3',

    keywords='statistics meteorology pre-processing testbed forecast '
             'assimilation data',

    packages=find_packages(exclude=['contrib', 'docs', 'tests.*', 'test']),

    test_suite='tests',

    scripts=glob.glob("bin/tfassim*"),
)
