# SCF Guess Tools

[![Conda](https://github.com/hauser-group/scf_guess_tools/actions/workflows/test.yaml/badge.svg)](https://github.com/hauser-group/scf_guess_tools/actions/workflows/test.yaml)

## About

[How to Contribute?](#how-to-contribute) This is a Python package providing a
uniform, high-level interface abstracting  common functionality from the
[Psi4](https://psicode.org) and [PySCF](https://pyscf.org) packages for:
- Calculating the electronic wavefunction of molecules using the Hartree-Fock 
method
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
`scf_guess_tools.psi` and `scf_guess_tools.py` backends. For a detailed
description of available features please refer to the in-source documentation.


```python
from scf_guess_tools import Backend, psi, py, load, guess, calculate
from scf_guess_tools import f_score, diis_error, energy_error

molecule = load("ch3.xyz", Backend.PY)  # or Backend.PSI
final = calculate(molecule, "pcseg-0")

for scheme in py.guessing_schemes:
    initial = guess(molecule, "pcseg-0", scheme)
    
    f = f_score(initial, final)
    print(f"f-score: {f}")
    
    d = diis_error(initial)
    print("DIIS error: {d")
    
    e = energy_error(initial, final)
    print("energy error: {e}")
```

### Caching

Caching of function results to disk is supported via `joblib.Memory`. To enable
it, you can pass `cache=True` to the `load`, `guess` and `calculate` functions.
You can also cache your own functions by applying the provided `@cache`
annotation.

```python
from scf_guess_tools import Backend, psi, load, guess, calculate, f_score
from scf_guess_tools import cache, clear_cache

molecule = load("hoclo.xyz", Backend.PSI, cache=True)
initial = guess(molecule, "pcseg-0", cache=True) # use default guessing scheme

# use wavefunction as initial guess
final = calculate(molecule, "pcseg-0", initial, cache=True)

@cache(ignore=["debug"]) # exclude debug from hash key
def score(initial: Wavefunction, final: Wavefunction, debug: bool) -> float:
    print(f"debug: {debug}")
    return f_score(initial, final)

f1 = score(initial, final, debug=True) # invokes the function
f2 = score(initial, final, debug=False) # returns cached result

clear_cache() # force clear cache
```

### Timing

Functions decorated with the `@timeable` annotation allow for tracking the
used CPU time as determined by `time.process_time()`. Passing in a `time=True`
parameter will cause the function to return a tuple `(result, time)`.

```python
from scf_guess_tools import Backend, load, calculate, f_score

molecule, load_time = load("hoclo.xyz", Backend.PSI, time=True)
print(f"loading took {load_time} s")

initial, guess_time = guess(molecule, "pcseg-0", time=True)
print(f"guessing took {guess_time} s")

final, calculate_time = calculate(molecule, "pcseg-0", initial, time=True)
print(f"calculating took {calculate_time} s")

f, score_time = f_score(initial, final, time=True)
print(f"scoring took {score_time} s")
```

## How to Contribute

Please create an [issue](https://github.com/hauser-group/scf_guess_tools/issues) and a corresponding [pull request](https://github.com/hauser-group/scf_guess_tools/pulls) with a
branch named `feature/fancy-new-feature`. Pull requests are eventually merged
into the `development` branch.

We're using pre-commit hooks and the [Black](https://github.com/psf/black) formatter to enforce a uniform
coding style. In order to set this up and to activate editable mode for seeing
live changes please run `make develop` after a [regular installation](#installation).
