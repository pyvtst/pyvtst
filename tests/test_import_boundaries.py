"""Importing pyvtst must not import third-party tools before they are used."""

import subprocess
import sys

import pytest

FORBIDDEN = ("ase", "rdkit", "pyscf", "geometric", "h5py")

# pyvtst.core (and the top-level package) never import a third-party tool.
CORE_SCRIPT = f"""
import importlib, pkgutil, sys
import pyvtst, pyvtst.units, pyvtst.core
for info in pkgutil.walk_packages(pyvtst.core.__path__, "pyvtst.core."):
    importlib.import_module(info.name)
print(",".join(m for m in {FORBIDDEN!r} if m in sys.modules))
"""

# Ports and adapter packages import PySCF, geomeTRIC and h5py only when an
# engine, optimizer or HDF5 store is built.
ADAPTER_SCRIPT = f"""
import importlib, pkgutil, sys
for name in ("pyvtst.ports", "pyvtst.engines", "pyvtst.optimizers", "pyvtst.stores"):
    pkg = importlib.import_module(name)
    for info in pkgutil.walk_packages(pkg.__path__, name + "."):
        importlib.import_module(info.name)
print(",".join(m for m in {FORBIDDEN!r} if m in sys.modules))
"""


@pytest.mark.parametrize(
    "script", [CORE_SCRIPT, ADAPTER_SCRIPT], ids=["core", "adapters"]
)
def test_imports_no_third_party_tools(script):
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


# Cache keys are pure: no ports, no adapters, no third-party tools.
KEYS_SCRIPT = f"""
import sys
import pyvtst.core.keys
layers = ("pyvtst.ports", "pyvtst.engines", "pyvtst.optimizers", "pyvtst.stores",
          "pyvtst.io", "pyvtst.reaction")
print(",".join(sorted(
    m for m in sys.modules
    if m.split(".")[0] in {FORBIDDEN!r} or m.startswith(layers)
)))
"""


def test_cache_keys_import_only_core():
    result = subprocess.run(
        [sys.executable, "-c", KEYS_SCRIPT], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
