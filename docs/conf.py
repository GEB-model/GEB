# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

import pypandoc

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "GEB"
with open("copyright.rst", "r") as f:
    copyright = f.read()
with open("authors.rst", "r") as f:
    author = f.read()

os.chdir("ODD")
output_folder = Path("../_build/html/ODD")
output_folder.mkdir(exist_ok=True, parents=True)
output = pypandoc.convert_file(
    "ODD_protocol.md",
    "pdf",
    outputfile=output_folder / "ODD_protocol.pdf",
    extra_args=[
        "--pdf-engine=xelatex",
        "-V",
        "geometry:margin=1.0in",
        "--citeproc",
        "--bibliography=../references.bib",
    ],
)
os.chdir("..")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",  # Napoleon is a Sphinx extension that enables Sphinx to parse both NumPy and Google style docstrings
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.autoprogram",
    "sphinxcontrib.autoyaml",
    "sphinxcontrib.bibtex",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
add_module_names = False
napoleon_custom_sections = [("Returns", "params_style")]
autoclass_content = "both"
autoyaml_level = 3
bibtex_bibfiles = ["references.bib"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
