#!/usr/bin/env make -f
# This Makefile is based on the Makefile within the repository:
# gitlab.com/nobodyinperson/python3-numericalmodel

SETUP.PY = ./setup.py
PACKAGE_FOLDER = tfassim
DOCS_FOLDER = docs
DOCS_API_FOLDER = docs/source/api
DOCS_HTML_FOLDER = docs/build/html
INIT.PY = $(shell find $(PACKAGE_FOLDER) -maxdepth 1 -type f -name '__init__.py')
RST_SOURCES = $(shell find $(DOCS_FOLDER) -type f -iname '*.rst')
PYTHON_SOURCES = $(shell find $(PACKAGE_FOLDER) -type f -iname '*.py')

VERSION = $(shell python -c "import tfassim;import sys;sys.stdout.write(tfassim.__version__)")

.PHONY: all
all: wheel docs

.PHONY: docs
docs: $(PYTHON_SOURCES) $(RST_SOURCES)
	cd $(DOCS_FOLDER) && make html

.PHONY: build
build:
	$(SETUP.PY) build

.PHONY: dist
dist:
	$(SETUP.PY) sdist

.PHONY: wheel
wheel:
	$(SETUP.PY) sdist bdist_wheel

.PHONY: upload
upload: wheel tag
	$(SETUP.PY) sdist upload -r pypi

.PHONY: upload-test
upload-test: wheel tag
	$(SETUP.PY) sdist upload -r pypitest

.PHONY: increase-patch
increase-patch: $(INIT.PY)
	perl -pi -e 's/(__version__\s*=\s*")(\d+)\.(\d+).(\d+)(")/$$1.(join ".",$$2,$$3,$$4+1).$$5/ge' $(INIT.PY)
	$(MAKE) $(SETUP.PY)

.PHONY: increase-minor
increase-minor: $(INIT.PY)
	perl -pi -e 's/(__version__\s*=\s*")(\d+)\.(\d+).(\d+)(")/$$1.(join ".",$$2,$$3+1,0).$$5/ge' $(INIT.PY)
	$(MAKE) $(SETUP.PY)

.PHONY: increase-major
increase-major: $(INIT.PY)
	perl -pi -e 's/(__version__\s*=\s*")(\d+)\.(\d+).(\d+)(")/$$1.(join ".",$$2+1,0,0).$$5/ge' $(INIT.PY)
	$(MAKE) $(SETUP.PY)
#
# write the INIT.PY version into setup.py
$(SETUP.PY): $(INIT.PY)
	perl -pi -e 's/^(.*__version__\s*=\s*")(\d+\.\d+.\d+)(".*)$$/$${1}$(VERSION)$${3}/g' $@

.PHONY: tag
tag:
	git tag -f v$(VERSION)

.PHONY: setup-test
setup-test:
	$(SETUP.PY) test

.PHONY: coverage
coverage:
	coverage run $(SETUP.PY) test
	coverage report
	coverage html

.PHONY: clean
clean: distclean

.PHONY: distclean
distclean:
	rm -rf *.egg-info
	rm -rf build
	rm -rf $$(find -type d -iname '__pycache__')
	rm -f $$(find -type f -iname '*.pyc')
	rm -rf htmlcov/
	rm -f .coverage
	rm -f $(gui_mofiles) $(gui_locale_potfile) $(python_locale_potfile) $(glade_locale_potfile)
	(cd $(DOCS_FOLDER) && make clean)

.PHONY: fulltest
fulltest: wheel docs coverage
