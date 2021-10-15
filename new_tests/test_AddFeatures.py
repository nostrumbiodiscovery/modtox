import os
import pandas as pd
from pandas.testing import assert_frame_equal
import rdkit

from modtox.modtox.new_models_classes.Features.add_feat import (
    AddGlide,
    AddTopologicalFingerprints,
    AddMordred,
)
from modtox.modtox.new_models_classes.Features.feat_enum import Features

from modtox.modtox.new_models_classes.mol import MoleculeFromChem

DATA = os.path.join(os.path.dirname(__file__), "data")
ACTIVES = os.path.join(DATA, "actives.sdf")
INACTIVES = os.path.join(DATA, "inactives.sdf")
GLIDE_CSV = os.path.join(DATA, "glide_features.csv")


def test_format_glide():
    g = AddGlide(
        "a", "b", "c", glide_csv=GLIDE_CSV
    )  # Create object to access staticmethod
    df = g.format_glide(g.glide_csv)
    model_df = pd.read_csv(os.path.join(DATA, "formatted_glide.csv"), index_col=0)
    assert_frame_equal(df, model_df)


def test_add_glide():
    class TestMol:
        def __init__(self, name) -> None:
            self.name = name

    mol1 = TestMol("sdf/actives_sanitized.sdf:1")
    mol2 = TestMol("sdf/actives_sanitized.sdf:2")
    mol3 = TestMol("X")

    g = AddGlide(mol1, mol2, mol3, glide_csv=GLIDE_CSV)
    ft_dict = g.calculate()
    assert list(ft_dict.keys()) == [mol1, mol2, mol3]

    assert len(ft_dict[mol1]) == len(ft_dict[mol2]) == len(ft_dict[mol3]) == 168

    assert any(val != 0 for val in list(ft_dict[mol1].values()))
    assert any(val != 0 for val in list(ft_dict[mol2].values()))
    assert all(val == 0 for val in list(ft_dict[mol3].values()))


def test_add_topo():
    class TestMol:
        def __init__(self, sm) -> None:
            self.molecule = rdkit.Chem.MolFromSmiles(sm)

    mol1 = TestMol("CCN(CC)CC")
    mol2 = TestMol("C=CC(CCC)C(C(C)C)CCC")

    g = AddTopologicalFingerprints(mol1, mol2)

    ft_dict = g.calculate()
    assert list(ft_dict.keys()) == [mol1, mol2]
    assert len(ft_dict[mol1]) == len(ft_dict[mol2]) == 2048
    assert all(
        isinstance(mol.topo, rdkit.DataStructs.cDataStructs.ExplicitBitVect)
        for mol in [mol1, mol2]
    )


def test_add_mordred():
    class TestMol:
        def __init__(self, sm) -> None:
            self.molecule = rdkit.Chem.MolFromSmiles(sm)

    mol1 = TestMol("CCN(CC)CC")
    mol2 = TestMol("C=CC(CCC)C(C(C)C)CCC")

    g = AddMordred(mol1, mol2)

    ft_dict = g.calculate()
    assert list(ft_dict.keys()) == [mol1, mol2]
    assert len(ft_dict[mol1]) == len(ft_dict[mol2]) == 1613
