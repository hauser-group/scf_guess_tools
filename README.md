# SCF Guess Tools

[![Conda](https://github.com/hauser-group/scf_guess_tools/actions/workflows/test.yaml/badge.svg)](https://github.com/hauser-group/scf_guess_tools/actions/workflows/test.yaml)

## About

This is a Python package ([How to Contribute](#how-to-contribute)) providing a
uniform, high-level interface abstracting  common functionality from the
[Psi4](https://psicode.org) and [PySCF](https://pyscf.org) packages for:
- Calculating the electronic wavefunction of molecules using the Hartree-Fock 
method
- Making initial guesses using classical guessing schemes
- Scoring arbitrary initial guesses with respect to impact on convergence

## How to Use

```python
from scf_guess_tools import Psi4Engine, PySCFEngine, Metric

engine = Psi4Engine() # you can replace this on-the-fly
molecule = engine.load("ch3.xyz")
final = engine.calculate(molecule, "pcseg-0")

for method in engine.guessing_schemes:
    initial = engine.guess(molecule, "pcseg-0", method)
    score = engine.score(initial, final, Metric.DIIS_ERROR)
    print(f"{method} scored {score}")
```

## Installation

This package is distributed via Conda. Currently, you need to manually build and
install this package as follows.

- Activate the conda environment to which you want to install
- Run `conda install conda-build`
- Run `conda build --channel conda-forge --channel pyscf recipe`
- Run `conda install --channel conda-forge --channel pyscf --use-local
scf_guess_tools`

In order to see live changes to the package immediately you need to install this
package in editable mode. After a regular installation, which is necessary to
ensure all dependencies are installed in the target Conda environment, please:

- Run `conda remove --force scf_guess_tools`
- Run `conda develop .`

Invoke `conda develop --uninstall .` to exit development mode.

## How to Contribute

Please create an [issue](https://github.com/hauser-group/scf_guess_tools/issues)
as well as a `feature/fancy-new-feature` branch linked to that issue. Feature
branches are merged into the `development` branch via
[pull requests](https://github.com/hauser-group/scf_guess_tools/pulls).

We're using pre-commit hooks and the [Black](https://github.com/psf/black)
formatter to enforce a uniform coding style. Before committing, please:
- Run `conda install pre-commit`
- Run `conda install black`
- Run `pre-commit install`