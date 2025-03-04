from pyscf.scf.uhf import UHF

from scf_guess_tools import (
    Metric,
    Molecule,
    Engine,
    PySCFEngine,
    Psi4Engine,
    Wavefunction,
)

import pandas as pd


def something(engine: Engine, scheme: str):
    basis = "pcseg-0"
    molecules = [
        "acetaldehyde.xyz",
        "CuMe.xyz",
        "hoclo.xyz",
    ]  ##"ch.xyz", "ch2-trip.xyz"
    molecules = {m.name: m for m in [engine.load(path) for path in molecules]}

    table = pd.DataFrame(index=molecules.keys(), columns=[Metric.DIIS_ERROR])

    for name in table.index:
        for metric in table.columns:
            molecule = molecules[name]
            initial = engine.guess(molecule, basis, scheme)
            final = engine.calculate(molecule, basis)
            table.at[name, metric] = engine.score(initial, final, metric)

    print(table)


# pd.options.display.float_format = "{:.3f}".format
something(PySCFEngine(cache=False), scheme="1e")  # , "1e")
something(Psi4Engine(cache=False), scheme="CORE")  # , "CORE")
exit()


engine = Psi4Engine(cache=True, verbose=1)
engine.memory.clear()


molecule = engine.load("CuMe.xyz")
# molecule.native.verbose = 0
first = engine.guess(molecule, basis, "CORE")

# self.molecule == other.molecule
#             and self.basis == other.basis
#             and self.initial == other.initial
#             and self.iterations == other.iterations
#             and self.retried == other.retried
#             and self.S == other.S
#             and self.F == other.F
#             and self.D == other.D

exit()

# def plot(labels, matrix):
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     # Convert to string format for plotting
#     ao_labels_str = [str(label) for label in labels]
#
#     # Plot overlap matrix with AO labels
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_aspect("equal")
#
#     p = ax.pcolormesh(matrix, cmap="bwr", shading="auto")
#     plt.colorbar(p)
#
#     # Set AO labels on x and y axis
#     ax.set_xticks(np.arange(len(labels)))  # Set tick positions
#     ax.set_yticks(np.arange(len(labels)))
#     ax.set_xticklabels(ao_labels_str, rotation=90, fontsize=8)  # Rotate x-labels
#     ax.set_yticklabels(ao_labels_str, fontsize=8)
#
#     # Adjust layout to prevent label overlap
#     plt.tight_layout()
#     plt.show()
#
#
# # pyengine = PySCFEngine(cache=False)
# # pymolecule = pyengine.load("hoclo.xyz")
# # pyfinal = pyengine.calculate(pymolecule, basis="sto-3g")
# # pylabels = pymolecule.native.ao_labels()
# # pyS = pyfinal.S
# # plot(pylabels, pyS)
#
# psiengine = Psi4Engine(cache=False)
# psimolecule = psiengine.load("hoclo.xyz")
# psifinal = psiengine.calculate(psimolecule, basis="sto-3g")
# basis = psifinal.native.basisset()
# ao_labels = []
# for s in range(psifinal.native.basisset().nshell()):
#     shell = psifinal.native.basisset().shell(s)
#     center = str(shell.ncenter+1)
#     am = shell.am
#     amchar = shell.amchar
#     basename = '{'+center+'}'+amchar
#     for j in range(0,am+1):
#         lx = am - j
#         for lz in range(0, j + 1):
#             ly  = j - lz
#             ao_labels.append(basename+'x'*lx+'y'*ly+'z'*lz)
# psilabels = ao_labels
# psiS = psifinal.S
# plot(psilabels, psiS)
#
#
# print()
# exit()

# print(molecule.singlet)
# #initial = engine.guess(molecule, basis="pcseg-0", method="CORE")
# final = engine.calculate(molecule, basis="pcseg-0")
# print("DONE")
# #score = engine.score(initial, final, Metric.DIIS_ERROR)
#
# #print(molecule.native.basis)
# #print(molecule.native.intor("int1e_ovlp"))
#
#
# first = final.D
# #print(first.shape)
# second = engine.calculate(molecule, basis="pcseg-0").D
# #print(second.shape)
#
# #print(first)
# print("EEEEEEEEEEEEEND")
# #print(second)
# print(first-second)


# @engine.memory.cache
# def getmolecule(mol):
#     return mol
#
# print(getmolecule(molecule).multiplicity)
# print(getmolecule(molecule).multiplicity)
