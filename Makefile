# Entry points. The Python is the interpreter of the owner's environment; set
# PYTHON=python on another machine.
PYTHON ?= ~/Workplace/python3.12/bin/python3.12
QUARTO ?= quarto

.PHONY: test docs docs-preview clean

test:          ## run every test (see pyproject.toml)
	$(PYTHON) -m pytest -n 8 -q -rs

docs:          ## build the API reference and the site from scratch, then check both (docs/_check_site.py)
	$(PYTHON) docs/_check_site.py --build --quarto $(QUARTO)

docs-preview:  ## render the site and serve it with live reload (no checks)
	cd docs && $(PYTHON) -m quartodoc build --config _quarto.yml
	cd docs && QUARTO_PYTHON=$(PYTHON) $(QUARTO) preview

clean:         ## remove the rendered site, the generated reference, the logs and the caches
	rm -rf docs/_site docs/reference docs/.quarto docs/_logs docs/objects.json docs/_objects.json examples/output
