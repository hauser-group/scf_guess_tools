from scf_guess_tools import Backend, calculate, load

backend = Backend.PY
mol = load("tests/molecules/geometries/ch3.xyz", backend)

final_hf = calculate(mol, "pcseg-1", "minao", "hf", None)
final_dft = calculate(mol, "pcseg-1", "minao", "dft", None)
