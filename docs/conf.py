"""Sphinx configuration for the pyvtst documentation."""

import re
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
    # Fixed-color copies of logo.svg; see _check_logo_variants.
    "logo": {
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
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


def _check_logo_variants() -> None:
    """Fail the build unless _static/logo-{light,dark,readme}.svg match logo.svg.

    logo.svg follows the system theme through a prefers-color-scheme query,
    which neither the docs' theme toggle nor GitHub's appearance setting can
    reach, so the docs use fixed-color copies of it. The README uses a single
    mid-tone copy because PyPI has no dark theme: a prefers-color-scheme
    switch there shows the dark logo on PyPI's white page.
    """
    docs = Path(__file__).parent
    svg = (docs / "logo.svg").read_text(encoding="utf-8")
    light = re.sub(
        r"\s*@media \(prefers-color-scheme: dark\) \{[^}]*\}\s*\}?", "\n", svg
    )
    if light == svg:
        raise ValueError("logo.svg: the prefers-color-scheme block was not found")
    inks = {"light": "#22223B", "dark": "#ECEAF4", "readme": "#77738F"}
    expected = {mode: light.replace("#22223B", ink) for mode, ink in inks.items()}
    for mode, text in expected.items():
        path = docs / "_static" / f"logo-{mode}.svg"
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(
                f"{path.name} differs from logo.svg; regenerate it after "
                f"editing logo.svg (it swaps #22223B for {inks[mode]})"
            )


def _build_assets(app: Any, config: Any) -> None:
    """Write the figures and plotly.js; check the logo variants."""
    docs = Path(__file__).parent
    sys.path.insert(0, str(docs))
    import figures  # noqa: PLC0415
    import plotly  # noqa: PLC0415

    figures.build(docs / "_generated" / "figures")
    static = docs / "_generated" / "static"
    static.mkdir(parents=True, exist_ok=True)
    bundle = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    shutil.copyfile(bundle, static / "plotly.min.js")
    _check_logo_variants()


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
