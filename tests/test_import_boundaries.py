"""pyvtst.core (and the top-level package) must not import third-party tools."""

import subprocess
import sys

FORBIDDEN = ("ase", "rdkit", "pyscf", "geometric", "h5py")

SCRIPT = f"""
import importlib, pkgutil, sys
import pyvtst, pyvtst.units, pyvtst.core
for info in pkgutil.walk_packages(pyvtst.core.__path__, "pyvtst.core."):
    importlib.import_module(info.name)
print(",".join(m for m in {FORBIDDEN!r} if m in sys.modules))
"""


def test_core_imports_no_third_party_tools():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
