.PHONY: test
test:
	@bash -c '\
	check_env() { \
	  var_name=$$1; \
	  subdir=$$2; \
	  eval "var_value=\$$$${var_name}"; \
	  if [ -z "$$var_value" ]; then \
	    export $$var_name="$$HOME/scratch/$$subdir"; \
	  fi; \
	  echo "$$var_name=$$var_value"; \
	  mkdir -p "$$var_value"; \
	}; \
	\
	check_env PSI_SCRATCH psi; \
	check_env PYSCF_TMPDIR pyscf; \
	check_env SGT_CACHE sgt; \
	cd tests; \
	pytest -vv'

