# SCF Guess Tools v1.0

[![Conda](https://github.com/hauser-group/scf_guess_tools/actions/workflows/test.yaml/badge.svg)](https://github.com/hauser-group/scf_guess_tools/actions/workflows/test.yaml)

## About

[How to Contribute?](#how-to-contribute) This is a Python package providing a
uniform, high-level interface abstracting  common functionality from the
[Psi4](https://psicode.org) and [PySCF](https://pyscf.org) packages for:
- Calculating the electronic wavefunction of molecules using the Hartree-Fock 
method or Density functional theory
- Making initial guesses using classical guessing schemes
- Scoring arbitrary initial guesses relative to the converged solution

## Installation

This package is distributed via Conda. Currently, you need to manually build and
install this package as follows:

- Activate the conda environment to which you want to install
- Run `make build`
- Run `make install`
- Run `make test` (optional)

## How to Use

### Environment Variables

You should specify the following environment variables:
- `SGT_CACHE`: directory to store cached function results (see
[Caching](#caching))
- `PSI_SCRATCH`: scratch directory for the `Psi` backend
- `PYSCF_TMPDIR`: scratch directory for the `Py` backend

### Core Functionality

This package provides uniform interfaces such as the abstract `Molecule`,
`Wavefunction` and `Matrix` classes. These interfaces are implemented by the
`scf_guess_tools.psi` and `scf_guess_tools.py` backends. For a more detailed
description of available features please refer to the in-source documentation.


```python
from scf_guess_tools import Backend, load, build, guess, calculate, guessing_schemes
from scf_guess_tools import f_score, diis_error, energy_error

backend = Backend.PY # or Backend.PSI

molecule = load("ch3.xyz", backend, symmetry=False)
print(f"molecule name: {molecule.name}")
print(f"molecule charge: {molecule.charge}")
print(f"molecule multiplicity: {molecule.multiplicity}")
print(f"molecule is singlet: {molecule.singlet}")

final = calculate(molecule, "pcseg-0")
print(f"solution converged: {final.converged}")
print(f"solution stable: {final.stable}")
print(f"solution required 2nd order scf: {final.second_order}")

# for dft calculations one has to specify method='dft' and a supported funcitonal
# due to the nature of dft some feature like fock matrix and therefore diis_error are not available
final_dft = calculate(molecule, "pcseg-0", method="dft", functional="b3lyp")

print(f"overlap: {final.overlap()}")

# For non-singlets the alpha and beta matrices are returned as a tuple
# For singlets, a single matrix is returned
# Setting tuplify=True ensures that a tuple is returned, regardless of multiplicity
for density in final.density(tuplify=True):
    print(f"density: {density}")

for scheme in guessing_schemes(backend):
    initial = guess(molecule, "pcseg-0", scheme)

    f = f_score(initial.overlap(), initial.density(), final.density())
    print(f"f-score: {f}")

    d = diis_error(initial.overlap(), initial.density(), initial.fock())
    print(f"DIIS error: {d}")

    e = energy_error(initial.electronic_energy(), final.electronic_energy())
    print(f"energy error: {e}")

import numpy as np

# We can also build matrices from arbitrary numpy arrays
original = initial.density(tuplify=True)[0]
array = np.random.rand(*original.numpy.shape)
matrix = build(array, backend)

f = f_score(initial.overlap(), matrix, final.density())
print(f"f-score of random unnormalized density: {f}")
```

#### DFT-Functionals
Auto-detection of dft-functionals strings is only possible as long as the backend supports the functional strings. Please refer to the respective documentation

Psi4: https://psicode.org/psi4manual/master/dft_byfunctional.html

PySCF: https://pyscf.org/user/dft.html#customizing-xc-functionals

If not specified 'B3LYP-VWN3' is used. 

### Caching

Caching of function results to disk is supported via `joblib.Memory`. To enable
it, you can pass `cache=True` to the `load`, `guess` and `calculate` functions.
You can also cache your own functions by applying the provided `@cache`
annotation.

Warning: `Molecule`, `Wavefunction` and `Matrix` instances might exhibit
floating  point fluctuations upon serialization and de-serialization. This can
cause semantically equivalent objects to hash to different keys. It is
recommended to use basic datatypes as parameters for cached functions.

```python
from scf_guess_tools import Backend, Wavefunction, load, guess, calculate, f_score
from scf_guess_tools import cache, clear_cache

molecule = load("hoclo.xyz", Backend.PSI, cache=True)
initial = guess(molecule, "pcseg-0", cache=True) # use default guessing scheme

# use wavefunction as initial guess
final = calculate(molecule, "pcseg-0", initial, cache=True)

@cache(ignore=["debug"]) # exclude debug from hash key
def score(initial: Wavefunction, final: Wavefunction, debug: bool) -> float:
    print(f"debug: {debug}")
    return f_score(initial.overlap(), initial.density(), final.density())

f1 = score(initial, final, debug=True) # invokes the function
f2 = score(initial, final, debug=False) # returns cached result

clear_cache() # force clear cache
```

### Timing

Functions decorated with the `@timeable` annotation allow for tracking the
used CPU time as determined by `time.process_time()`. Passing in a `time=True`
parameter will cause the function to return a tuple `(result, time)`.

```python
from scf_guess_tools import Backend, load, guess, calculate, f_score

molecule, load_time = load("hoclo.xyz", Backend.PSI, time=True)
print(f"loading took {load_time} s")

initial, guess_time = guess(molecule, "pcseg-0", time=True)
print(f"guessing took {guess_time} s")

final, calculate_time = calculate(molecule, "pcseg-0", initial, cache=True, time=True)
print(f"calculating took {calculate_time} s")

final, calculate_time = calculate(molecule, "pcseg-0", initial, cache=True, time=True)
print(f"cached calculation took {calculate_time} s")

f, score_time = f_score(initial.overlap(), initial.density(), final.density(), time=True)
print(f"scoring took {score_time} s")
```

## How to Contribute

Please create an [issue](https://github.com/hauser-group/scf_guess_tools/issues) and a corresponding [pull request](https://github.com/hauser-group/scf_guess_tools/pulls) with a
branch named `feature/fancy-new-feature`. Pull requests are eventually merged
into the `development` branch.

We're using pre-commit hooks and the [Black](https://github.com/psf/black) formatter to enforce a uniform
coding style. In order to set this up and to activate editable mode for seeing
live changes please run `make develop` after a [regular installation](#installation).
