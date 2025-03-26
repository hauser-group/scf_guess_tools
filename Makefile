CONDA_BUILD := $(shell echo $$CONDA_PREFIX)/conda-bld

build:
	conda install --yes conda-build
	conda build --channel conda-forge --channel pyscf recipe

install:
	conda install --yes --channel conda-forge --channel pyscf --channel $(CONDA_BUILD) --use-local scf_guess_tools

develop:
	conda install --yes --channel conda-forge pre-commit black
	pre-commit install
	conda remove --yes --force scf_guess_tools
	conda develop .

test:
	conda install --yes --channel conda-forge pytest python-constraint
	@bash -c '\
	set_env() { \
	  local random="$$(uuidgen)"; \
	  local path="$$HOME/sgt-scratch-$$2-$$random"; \
	  mkdir -p "$$path"; \
	  export $$1="$$HOME/sgt-scratch-$$2-$$random"; \
	  eval "value=\$$$${1}"; \
	  echo "$$1=$$value"; \
	}; \
	set_env PSI_SCRATCH psi; \
	set_env PYSCF_TMPDIR py; \
	set_env SGT_CACHE sgt; \
	cd tests; \
	pytest -vv;'

.PHONY: build install develop test