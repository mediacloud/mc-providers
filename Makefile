# from es-tools

# to create development environment: `make`
# to run pre-commit linting/formatting: `make lint`

VENVDIR=venv
VENVBIN=$(VENVDIR)/bin
VENVDONE=$(VENVDIR)/.done

help:
	@echo Usage:
	@echo "make install -- installs pre-commit hooks, dev environment"
	@echo "make lint -- runs pre-commit checks"
	@echo "make test-mc -- run MC news provider tests"
	@echo "make test-all -- run all tests"
	@echo "make update -- update .pre-commit-config.yaml"
	@echo "make clean -- remove development environment"

## run pre-commit checks on all files
lint:	$(VENVDONE)
	$(VENVBIN)/pre-commit run --all-files

# create venv with project dependencies
# --editable skips installing project sources in venv
# pre-commit is in dev optional-requirements
install $(VENVDONE): $(VENVDIR) Makefile pyproject.toml
	$(VENVBIN)/python3 -m pip install --editable '.[dev,test]'
	$(VENVBIN)/pre-commit install
	touch $(VENVDONE)

$(VENVDIR):
	python3 -m venv $(VENVDIR)

## update .pre-commit-config.yaml
update:	$(VENVDONE)
	$(VENVBIN)/pre-commit autoupdate

test-all:	$(VENVDONE)
	@test -n "$$MEDIA_CLOUD_API_KEY" || (echo "need MEDIA_CLOUD_API_KEY" 1>&2 && exit 1)
	$(VENVBIN)/pytest

test-mc:	$(VENVDONE)
	@test -n "$$MEDIA_CLOUD_API_KEY" || (echo "need MEDIA_CLOUD_API_KEY" 1>&2 && exit 1)
	$(VENVBIN)/pytest -v mc_providers/test/test_onlinenews.py::OnlineNewsMediaCloudProviderTest

## clean up development environment
clean:
	-$(VENVBIN)/pre-commit clean
	rm -rf $(VENVDIR) build *.egg-info .pre-commit-run.sh.log \
		__pycache__ .mypy_cache
