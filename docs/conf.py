# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'space-coe'
copyright = '2026, HPAI-BSC'
author = 'HPAI-BSC'
release = '0.0.1'

# -- General configuration ----------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',      # Google-style docstrings (src/data_generation)
    'sphinx.ext.viewcode',      # "view source" links
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
    'myst_parser',              # allow Markdown pages alongside .rst
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Napoleon: this repo mixes Google-style docstrings (src/data_generation)
# with plain reST `:param:`/`:return:` docstrings (src/models) — both are
# valid Sphinx/napoleon input, no extra config needed for the latter.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}
autodoc_typehints = 'description'

# Third-party/optional deps not needed to render docs (heavy, HPC-specific,
# or not installed in every environment) — mocked so autodoc can still
# import and document the modules that reference them.
autodoc_mock_imports = [
    'mamba_ssm',
    'selective_scan_cuda',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Docstrings across this repo use markdown-style single backticks (`x`) for
# inline code, which plain reST parses as an unresolvable role reference
# rather than a code span. Rendering single backticks as inline literals
# (like reST's double backticks) avoids touching every docstring to
# convert them.
default_role = 'literal'

# -- Options for HTML output --------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_theme_options = {
    'light_logo': 'space-logo.png',
    'dark_logo': 'space-logo-dark.png',
}

# -- Regenerating docs/api -----------------------------------------------------
# Run from the repo root after adding/renaming Python modules. The custom
# templates in docs/_templates/apidoc drop the leading "src." from page titles.
#
#   ./venv/bin/sphinx-apidoc -f -e --implicit-namespaces \
#       --templatedir=docs/_templates/apidoc -o docs/api src \
#       "src/data" "src/data_generation/venv" "src/data_generation/original_scripts" \
#       "src/data_generation/test" "src/models/components_old" \
#       "src/*/test" "src/*/tests"
#
# Build:  rm -rf docs/_build && ./venv/bin/sphinx-build -b html docs docs/_build/html
