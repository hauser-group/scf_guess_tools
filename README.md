# SCF Guess Tools

[![Conda](https://github.com/hauser-group/scf_guess_tools/actions/workflows/test.yaml/badge.svg)](https://github.com/hauser-group/scf_guess_tools/actions/workflows/test.yaml)

## About

This is a Python package providing a uniform, high-level interface abstracting
common functionality from the [Psi4](https://psicode.org) and
[PySCF](https://pyscf.org) packages for:
- Calculating the electronic wavefunction of molecules using the Hartree-Fock 
method
- Making initial guesses using classical guessing schemes
- Scoring arbitrary initial guesses with respect to impact on convergence

## Installation

This package is distributed via Conda. Currently, you need to manually build and
install this package as follows.

- Activate the conda environment to which you want to install
- Run `conda install conda-build`
- Run `conda build --channel conda-forge --channel pyscf recipe`
- Run `conda install --channel conda-forge --channel pyscf --use-local
scf_guess_tools`

## Development

To install the package in editable mode, first do a regular installation to the
target conda environment to ensure all dependencies are available. Then:
- Run `conda remove --force scf_guess_tools`
- Run `conda develop .`

Invoke `conda develop --uninstall .` to exit development mode.
