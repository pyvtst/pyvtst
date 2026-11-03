"""Sphinx configuration for the pyvtst documentation."""

from importlib.metadata import version as _version

project = "pyvtst"
author = "Emilian Roșca"
copyright = "%Y, Emilian Roșca"
release = _version("pyvtst")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

# API reference: one page per module, generated on every build.
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    # Document each object once, where it is defined, not again in every
    # package that re-exports it.
    "ignore-module-all": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
typehints_use_signature = False
typehints_use_signature_return = False
always_use_bars_union = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False

# README.md links to its own sections by heading anchor.
myst_heading_anchors = 4
myst_enable_extensions = ["dollarmath"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

html_theme = "furo"
html_title = f"pyvtst {release}"
html_logo = "logo.svg"
html_theme_options = {
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/pyvtst/pyvtst",
    "source_branch": "main",
    "source_directory": "docs/",
}
