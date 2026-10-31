"""3D starting geometries from SMILES via RDKit. RDKit is imported lazily."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import numpy.typing as npt

from pyvtst.core.molecule import Molecule
from pyvtst.core.units import BOHR_TO_ANGSTROM


def _require_rdkit() -> tuple[ModuleType, ModuleType, ModuleType]:
    # RDKit is optional, so it is imported on first use rather than at load time.
    try:
        from rdkit import Chem  # noqa: PLC0415
        from rdkit.Chem import AllChem, Descriptors  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            'pyvtst.io.from_smiles needs RDKit: pip install "pyvtst[rdkit]"'
        ) from exc
    return Chem, AllChem, Descriptors


def from_smiles(
    smiles: str,
    charge: int | None = None,
    mult: int | None = None,
    n_conformers: int = 10,
    seed: int = 0xF00D,
    masses: npt.ArrayLike | None = None,
    symmetry_number: int = 1,
    name: str | None = None,
) -> Molecule:
    """Generate a 3D starting geometry from SMILES. Needs ``pyvtst[rdkit]``.

    Embeds ``n_conformers`` ETKDG conformers, pre-optimizes them with MMFF94
    when it has parameters for the molecule, otherwise UFF, otherwise leaves
    them as embedded, and returns the lowest-energy one. Charge defaults to the
    sum of formal charges, multiplicity to radical electrons + 1 and the name
    to the SMILES. The result is a guess for later optimization, for stable
    species (minima) only.
    """
    Chem, AllChem, Descriptors = _require_rdkit()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES {smiles!r}")
    if charge is None:
        charge = Chem.GetFormalCharge(mol)
    if mult is None:
        mult = Descriptors.NumRadicalElectrons(mol) + 1
    mol = Chem.AddHs(mol)

    if mol.GetNumAtoms() == 1:
        positions = np.zeros((1, 3))
    else:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        )
        if not conf_ids:
            raise RuntimeError(
                f"could not embed a 3D conformer for SMILES {smiles!r}; "
                "try raising n_conformers or changing seed"
            )
        if AllChem.MMFFHasAllMoleculeParams(mol):
            results = AllChem.MMFFOptimizeMoleculeConfs(mol)
        elif AllChem.UFFHasAllMoleculeParams(mol):
            results = AllChem.UFFOptimizeMoleculeConfs(mol)
        else:
            results = None
        if results:
            # Lowest energy wins; ties go to the lower conformer id.
            _, best = min(
                (energy, cid)
                for cid, (_, energy) in zip(conf_ids, results, strict=True)
            )
        else:
            best = conf_ids[0]
        positions = mol.GetConformer(best).GetPositions()

    return Molecule(
        symbols=tuple(atom.GetSymbol() for atom in mol.GetAtoms()),
        coords=positions / BOHR_TO_ANGSTROM,
        charge=charge,
        mult=mult,
        masses=masses,
        symmetry_number=symmetry_number,
        name=smiles if name is None else name,
    )
