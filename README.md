# SCF Guess Tools

[![Conda](https://github.com/hauser-group/scf_guess_tools/actions/workflows/test.yaml/badge.svg)](https://github.com/hauser-group/scf_guess_tools/actions/workflows/test.yaml)

## About

[How to Contribute?](#how-to-contribute) This is a Python package providing a
uniform, high-level interface abstracting  common functionality from the
[Psi4](https://psicode.org) and [PySCF](https://pyscf.org) packages for:
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

If you run into issues with the last conda install command, you can try: `conda install --channel conda-forge --channel pyscf --channel <~/path to conda build folder (e.g. ~/miniconda3/envs/<cond_env_name>/conda-bld/)> scf_guess_tools`. 

## How to Use

### Scratch Directories

You should specify scratch directories used by the backends via the
`PSI_SCRATCH` and `PYSCF_TMPDIR` environment variables.

### Core Functionality

The `Engine` interface as implemented by `Psi4Engine` and `PySCFEngine` is the
core of this package. It provides a common API for loading molecular geometries,
making initial density guesses, calculating the converged density and scoring
guesses.

```python
from scf_guess_tools import Metric, Psi4Engine, PySCFEngine

engine = Psi4Engine()  # you can switch between engines on-the-fly
molecule = engine.load("ch3.xyz")
final = engine.calculate(molecule, "pcseg-0")

for scheme in engine.guessing_schemes():
    initial = engine.guess(molecule, "pcseg-0", scheme)
    score = engine.score(initial, final, Metric.DIIS_ERROR)
    print(f"{scheme} scored {score}")
```

### Caching

This package supports caching of calculated results to disk. To enable this,
specify the cache directory by setting the `SGT_CACHE` environment variable and
use `cache=True` when constructing an `Engine`. You can even cache custom
functions by using the `@Engine.memory.cache` annotation.

```python
from scf_guess_tools import Metric, Molecule, Psi4Engine, Wavefunction

engine = Psi4Engine(cache=True, verbose=1)
molecule = engine.load("hoclo.xyz")
initial = engine.guess(molecule, basis="pcseg-0", scheme="CORE")  # will be cached
final = engine.calculate(molecule, basis="pcseg-0")  # will be cached
score = engine.score(initial, final, Metric.DIIS_ERROR)  # will be cached

@engine.memory.cache(verbose=1, ignore=["debug"])
def get_molecule(wavefunction: Wavefunction, debug: bool) -> Molecule:
    print(f"debug: {debug}")
    return wavefunction.molecule

final_molecule = get_molecule(final, debug=True)  # invokes get_molecule
get_molecule(final, debug=False)  # returns cached result

engine.memory.clear()  # clears the cache
```

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

In order to see live changes to the package immediately, you need to install
this  package in editable mode. After a [regular installation](#installation),
which is necessary to ensure all dependencies are installed in the target Conda environment, in the repository root please:

- Run `conda remove --force scf_guess_tools`
- Run `conda develop .`

Invoke `conda develop --uninstall .` to exit development mode.