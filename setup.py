#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "model-tools @ git+https://github.com/brain-score/model-tools",
    "tensorflow",  # authors used 1.14
    "keras",  # 2.2.4
    "scikit-image",  # 0.17.2
    "h5py",  # 2.9.0
]

setup(
    name='u-pred-net',
    version='0.1.0',
    url='https://github.com/uranc/U-Net-Pred',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
)
