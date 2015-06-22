#!/usr/bin/env python

from setuptools import setup

setup(
    name = 'tftools',
    version = '0.1.0',
    description = 'Utilities for optimizing transfer function excitation signals',
    author = 'Larry Price',
    author_email = 'larry.r.price@gmail.com',
    url = 'http://github.com/lrp/tftools',
    license = 'GNU GPLv3',
    py_modules = ['tftools'],
    install_requires = ['numpy>=1.91', 'scipy>=0.15.1', 'matplotlib>=1.4.2']
)
