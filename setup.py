#!/usr/bin/env python3
""" acoustic_gps is a package for sound field reconstruction using 
Gaussian processes.

It provides:

- complex valued Gaussian process regression for sound field 
  reconstruction
- kernels based on both radial basis functions and plane waves 
  expansions
- Bayesian inference of the kernel hyperparameters via pyStan
- sound field visualization tools
- and more

Besides its obvious use for acoustics, the kernels in acoustic_gps can 
be easily adapted for the reconstruction of other fields such as 
electromagnetic.
"""
from distutils.core import setup  # nopep8
setup(
    author="Diego Caviedes Nozal",
    name="acoustic_gps",
    version="0.1",
    packages=["acoustic_gps"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    	"pystan",
        "pyDOE"
    ],
)
