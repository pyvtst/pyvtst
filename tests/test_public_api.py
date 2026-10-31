import pyvtst
import pyvtst.io


def test_io_functions_reexported_at_top_level():
    for name in ("read_xyz", "write_xyz", "from_ase", "to_ase", "from_smiles"):
        assert getattr(pyvtst, name) is getattr(pyvtst.io, name)
