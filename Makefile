.PHONY: test
test:
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
