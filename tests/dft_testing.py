from scf_guess_tools import Backend, calculate, load


backend = Backend.PY
mol = load("/home/etschgi1/scf_guess_tools/tests/molecules/geometries/ch4.xyz", backend)


def printer(res):
    print(f"solution converged: {res.converged}")
    print(f"solution stable: {res.stable}")
    print(f"solution required 2nd order scf: {res.second_order}")
    print(f"Energy: {res.electronic_energy()}")


final_hf = calculate(mol, "6-31G(d)", None, "hf", None)
final_dft = calculate(mol, "6-31G(d)", None, "dft", "b3lypg")
print("HF")
printer(final_hf)
print("DFT")
printer(final_dft)

backend = Backend.PSI
mol = load("/home/etschgi1/scf_guess_tools/tests/molecules/geometries/h2o.xyz", backend)
final_hf = calculate(mol, "6-31G(d)", None, "hf", None)
final_dft = calculate(mol, "6-31G(d)", None, "dft", "b3lyp")
print("HF")
printer(final_hf)
print("DFT")
printer(final_dft)
