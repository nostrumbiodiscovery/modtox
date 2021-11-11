from modtox.modtox.Molecules.col import CollectionFromSDF
from modtox.modtox.utils.enums import Features
from modtox.modtox.Molecules.mol import BaseMolecule, MoleculeFromChem

import os

from unittest.mock import MagicMock, patch
import pytest

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REDUCED_DATASET = os.path.join(DATA, "reduced_dataset")
FULL_DATASET = os.path.join(DATA, "full_dataset")

ACTIVES = os.path.join(REDUCED_DATASET, "actives.sdf")
INACTIVES = os.path.join(REDUCED_DATASET, "inactives.sdf")
GLIDE_CSV = os.path.join(REDUCED_DATASET, "glide_features.csv")

OUTPUT_FOLDER = os.path.join(DATA, "tests_outputs")

col = CollectionFromSDF(ACTIVES, INACTIVES)
col.add_features(Features.glide, glide_csv=GLIDE_CSV)
mol_w_glide = col.molecules[0]
mol_wo_glide = col.molecules[5]


def test_read_sdf():
    """Tests reading SDF files"""
    cole = CollectionFromSDF(ACTIVES, INACTIVES)
    assert len(cole.actives) == 9
    assert len(cole.inactives) == 16
    assert len(cole.actives) + len(cole.inactives) == len(cole.molecules)



@patch(
    "modtox.modtox.Molecules.mol.BaseMolecule.add_cluster_info", autospec=True
)
def test_cluster_topo(mock):
    """Tests clustering molecules by topological fingerprints.
    Asserts adding cluster info to all molecules.
    """
    ACTIVES = os.path.join(FULL_DATASET, "actives.sdf")
    INACTIVES = os.path.join(FULL_DATASET, "inactives.sdf")
    col = CollectionFromSDF(ACTIVES, INACTIVES)
    col.cluster(cutoff=0.3, fp="topological")
    assert len(col.clusters) == 159
    assert mock.call_count == 695
    assert all(hasattr(mol, "topo") for mol in col.molecules)


@patch(
    "modtox.modtox.Molecules.mol.BaseMolecule.add_cluster_info", autospec=True
)
def test_cluster_morgan(mock):
    """Tests clustering molecules by topological fingerprints.
    Asserts adding cluster info to all molecules.
    """
    ACTIVES = os.path.join(FULL_DATASET, "actives.sdf")
    INACTIVES = os.path.join(FULL_DATASET, "inactives.sdf")
    col = CollectionFromSDF(ACTIVES, INACTIVES)
    col.cluster(cutoff=0.3, fp="morgan")
    assert len(col.clusters) == 283
    assert mock.call_count == 695
    assert all(hasattr(mol, "morgan") for mol in col.molecules)


def test_cluster_error():
    """Tests error is raised if fp not "morgan" or "topological". 
    """
    ACTIVES = os.path.join(FULL_DATASET, "actives.sdf")
    INACTIVES = os.path.join(FULL_DATASET, "inactives.sdf")
    col = CollectionFromSDF(ACTIVES, INACTIVES)
    with pytest.raises(NotImplementedError):
        col.cluster(cutoff=0.3, fp="a")


