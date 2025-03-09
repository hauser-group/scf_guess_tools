.PHONY: test
test:
	@bash -c '\
	check_env() { \
	  eval "value=\$$$${1}"; \
	  if [ -z "$$value" ]; then \
	    export $$1="$$HOME/scratch/$$2"; \
	  fi; \
	  eval "value=\$$$${1}"; \
	  mkdir -p "$$value"; \
	  echo "$$1=$$value"; \
	}; \
	check_env PSI_SCRATCH psi; \
	check_env PYSCF_TMPDIR pyscf; \
	check_env SGT_CACHE sgt; \
	cd tests; \
	pytest -vv;'
