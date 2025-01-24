# SCF Guess Tools

[![Conda](https://github.com/hauser-group/scf_guess_tools/actions/workflows/test.yml/badge.svg)](https://github.com/hauser-group/scf_guess_tools/actions/workflows/test.yml)

## About

This is a Python package containing a collection of tools to work with initial
guesses as required by various SCF methods. It aims at abstracting common
functionality while supporting both `Psi4` and `PySCF` as backends.

## Installation

Currently you need to manually install this package as follows.

- Activate the conda environment to which you want to install
- Run `conda build --channel conda-forge --channel pyscf recipe`
- Run `conda install --channel conda-forge --channel pyscf --use-local
scf_guess_tools`

## Development

To install the package in editable mode, first do a regular installation to the
target conda environment to ensure all dependencies are available. Then:
- Run `conda remove --force scf_guess_tools`
- Run `conda develop .`

To exit development mode, invoke `conda develop --uninstall .`.
