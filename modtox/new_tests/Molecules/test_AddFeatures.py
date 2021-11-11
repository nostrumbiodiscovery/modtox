from modtox.modtox.Molecules.add_feat import AddGlide
from modtox.modtox.Molecules.add_feat import AddTopologicalFingerprints
from modtox.modtox.Molecules.add_feat import AddMorganFingerprints
from modtox.modtox.Molecules.add_feat import AddMordred

import os
import pandas as pd
import rdkit

from pandas.testing import assert_frame_equal

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REDUCED_DATASET = os.path.join(DATA, "reduced_dataset")

ACTIVES = os.path.join(REDUCED_DATASET, "actives.sdf")
INACTIVES = os.path.join(REDUCED_DATASET, "inactives.sdf")
GLIDE_CSV = os.path.join(REDUCED_DATASET, "glide_features.csv")

OUTPUT_FOLDER = os.path.join(DATA, "tests_outputs")


def test_format_glide():
    """Formats glide and compares to model dataframe."""
    g = AddGlide(
        ["mol1", "mol2", "mol3"], glide_csv=GLIDE_CSV
    )  # Create object to access staticmethod
    df = g.format_glide(g.glide_csv)
    model_df = pd.read_csv(
        os.path.join(OUTPUT_FOLDER, "formatted_glide.csv"), index_col=0
    )
    # df.to_csv(os.path.join(OUTPUT_FOLDER, "formatted_glide.csv"))
    assert_frame_equal(df, model_df)


def test_add_glide():
    """Tests adding glide features to molecules.
    1. All columns start with "glide_"
    2. All molecules have glide features
    3. Matching molecules have glide features !=0
    4. Non matchin molecules (mol3) glide features == 0
    5. There are 168 glide features
    """
    class TestMol:
        def __init__(self, name) -> None:
            self.name = name

    mol1 = TestMol("sdf/actives_sanitized.sdf:1")
    mol2 = TestMol("sdf/actives_sanitized.sdf:2")
    mol3 = TestMol("X")

    g = AddGlide([mol1, mol2, mol3], glide_csv=GLIDE_CSV)
    ft_dict = g.calculate()
    assert all(
        col.startswith("glide")
        for glide_dict in ft_dict.values()
        for col in glide_dict.keys()
    )
    assert list(ft_dict.keys()) == [mol1, mol2, mol3]

    assert (
        len(ft_dict[mol1]) == len(ft_dict[mol2]) == len(ft_dict[mol3]) == 168
    )

    assert any(val != 0 for val in list(ft_dict[mol1].values()))
    assert any(val != 0 for val in list(ft_dict[mol2].values()))
    assert all(val == 0 for val in list(ft_dict[mol3].values()))


def test_add_topo():
    """Tests adding topological fingerprints to molecules.
    1. There are 2048 features.
    2. Stores fingerprints in mol.topo attribute.
    3. All features start with "topo_"
    4. Features are calculated and added for all molecules.
    """
    class TestMol:
        def __init__(self, sm) -> None:
            self.molecule = rdkit.Chem.MolFromSmiles(sm)

    mol1 = TestMol("CCN(CC)CC")
    mol2 = TestMol("C=CC(CCC)C(C(C)C)CCC")

    g = AddTopologicalFingerprints([mol1, mol2])

    ft_dict = g.calculate()
    assert list(ft_dict.keys()) == [mol1, mol2]
    assert len(ft_dict[mol1]) == len(ft_dict[mol2]) == 2048
    assert all(
        isinstance(mol.topo, rdkit.DataStructs.cDataStructs.ExplicitBitVect)
        for mol in [mol1, mol2]
    )
    assert all(
        col.startswith("topo")
        for glide_dict in ft_dict.values()
        for col in glide_dict.keys()
    )


def test_add_morgan():
    """Tests adding topological fingerprints to molecules.
    1. There are 2048 features.
    2. Stores fingerprints in mol.topo attribute.
    3. All features start with "morgan_"
    4. Features are calculated and added for all molecules.
    """
    class TestMol:
        def __init__(self, sm) -> None:
            self.molecule = rdkit.Chem.MolFromSmiles(sm)

    mol1 = TestMol("CCN(CC)CC")
    mol2 = TestMol("C=CC(CCC)C(C(C)C)CCC")

    g = AddMorganFingerprints([mol1, mol2])

    ft_dict = g.calculate()
    assert list(ft_dict.keys()) == [mol1, mol2]
    assert len(ft_dict[mol1]) == len(ft_dict[mol2]) == 1024
    assert all(
        isinstance(mol.morgan, rdkit.DataStructs.cDataStructs.ExplicitBitVect)
        for mol in [mol1, mol2]
    )
    assert all(
        col.startswith("morgan")
        for glide_dict in ft_dict.values()
        for col in glide_dict.keys()
    )


def test_add_mordred():
    """Tests adding topological fingerprints to molecules.
    1. There are 2048 features.
    2. Stores fingerprints in mol.topo attribute.
    3. All features start with "morgan_"
    4. Features are calculated and added for all molecules.
    """

    class TestMol:
        def __init__(self, sm) -> None:
            self.molecule = rdkit.Chem.MolFromSmiles(sm)

    mol1 = TestMol("CCN(CC)CC")
    mol2 = TestMol("C=CC(CCC)C(C(C)C)CCC")

    g = AddMordred([mol1, mol2])

    ft_dict = g.calculate()
    assert list(ft_dict.keys()) == [mol1, mol2]
    assert len(ft_dict[mol1]) == len(ft_dict[mol2]) == 1613
    assert all(
        col.startswith("mordred")
        for mord_dict in ft_dict.values()
        for col in mord_dict.keys()
    )
