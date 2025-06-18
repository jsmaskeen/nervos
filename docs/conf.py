# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
sys.path.insert(0,os.path.abspath(".."))

import shutil

def copy_example_notebooks():
    docs_path = os.path.dirname(__file__)
    dest = os.path.join(docs_path, "notebooks")
    src = os.path.abspath(os.path.join(docs_path, "../lib_examples"))

    os.makedirs(dest, exist_ok=True)

    notebook_files = ["circles.ipynb", "iris.ipynb", "mnist.ipynb"]

    for nb in notebook_files:
        src_file = os.path.join(src, nb)
        dest_file = os.path.join(dest, nb)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_file)

copy_example_notebooks()

project = 'nervos'
copyright = '2025, Jaskirat Singh Maskeen'
author = 'Jaskirat Singh Maskeen'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc","sphinx.ext.todo", "sphinx.ext.viewcode","sphinx.ext.coverage","sphinx.ext.napoleon","nbsphinx","myst_parser","sphinx.ext.autosummary"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','**.ipynb_checkpoints','**/storage/**']
nbsphinx_execute = 'never'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
