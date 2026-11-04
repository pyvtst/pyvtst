"""Sphinx configuration for the pyvtst documentation."""

import shutil
import sys
from importlib.metadata import version as _version
from pathlib import Path
from typing import Any

project = "pyvtst"
author = "Emilian Roșca"
copyright = "%Y, PyVTST Developers"
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

# User guide pages link to their own sections by heading anchor.
myst_heading_anchors = 4
myst_enable_extensions = ["dollarmath"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_title = f"PyVTST v{release} documentation"
html_theme_options = {
    # One logo for both themes, and the same file the README shows on GitHub
    # and PyPI: the teal wordmark reads 3.7:1 on white and 5:1 on dark.
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
    },
    "github_url": "https://github.com/pyvtst/pyvtst",
    "use_edit_page_button": True,
    "navbar_align": "left",
    "show_toc_level": 2,
}
# Only the User Guide has a left sidebar.
html_sidebars = {
    page: []
    for page in ("index", "quickstart", "api", "contributing", "_autosummary/*")
}
html_static_path = ["_static", "_generated/static"]
# Target of the "Edit on GitHub" button.
html_context = {
    "github_user": "pyvtst",
    "github_repo": "pyvtst",
    "github_version": "main",
    "doc_path": "docs",
}


def _build_assets(app: Any, config: Any) -> None:
    """Write the figures and plotly.js."""
    docs = Path(__file__).parent
    sys.path.insert(0, str(docs))
    import figures  # noqa: PLC0415
    import plotly  # noqa: PLC0415

    figures.build(docs / "_generated" / "figures")
    static = docs / "_generated" / "static"
    static.mkdir(parents=True, exist_ok=True)
    bundle = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    shutil.copyfile(bundle, static / "plotly.min.js")


def _add_figure_scripts(
    app: Any, pagename: str, templatename: str, context: dict[str, Any], doctree: Any
) -> None:
    """Load plotly.js only on pages that have figures."""
    if "pyvtst-figure" in context.get("body", ""):
        app.add_js_file("plotly.min.js", loading_method="defer")
        app.add_js_file("figures.js", loading_method="defer")


def setup(app: Any) -> None:
    # Before Sphinx checks that html_static_path exists.
    app.connect("config-inited", _build_assets, priority=100)
    app.connect("html-page-context", _add_figure_scripts)
