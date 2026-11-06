"""``python -m pyvtst INPUT``: the same as the ``pyvtst`` command."""

import sys

from pyvtst.inputfile.cli import main

if __name__ == "__main__":
    sys.exit(main())
