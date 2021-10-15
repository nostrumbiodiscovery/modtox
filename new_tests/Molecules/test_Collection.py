from modtox.modtox.Molecules.col import CollectionFromSDF
from modtox.modtox.utils.enums import Features

import os

from unittest.mock import patch
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
    cole = CollectionFromSDF(ACTIVES, INACTIVES)
    assert len(cole.actives) == 9
    assert len(cole.inactives) == 16
    assert len(cole.actives) + len(cole.inactives) == len(cole.molecules)

@patch('modtox.modtox.Molecules.mol.BaseMolecule.add_cluster_info', autospec=True)
def test_cluster_topological(mock):
    ACTIVES = os.path.join(FULL_DATASET, "actives.sdf")
    INACTIVES = os.path.join(FULL_DATASET, "inactives.sdf")
    col = CollectionFromSDF(ACTIVES, INACTIVES)
    col.cluster(cutoff=0.3, fp="topological")
    assert len(col.clusters) == 159
    assert mock.call_count == 695

@patch('modtox.modtox.Molecules.mol.BaseMolecule.add_cluster_info', autospec=True)
def test_cluster_morgan(mock):
    ACTIVES = os.path.join(FULL_DATASET, "actives.sdf")
    INACTIVES = os.path.join(FULL_DATASET, "inactives.sdf")
    col = CollectionFromSDF(ACTIVES, INACTIVES)
    col.cluster(cutoff=0.3, fp="morgan")
    assert len(col.clusters) == 283
    assert mock.call_count == 695

@patch('modtox.modtox.Molecules.mol.BaseMolecule.add_cluster_info', autospec=True)
def test_cluster_error(mock):
    ACTIVES = os.path.join(FULL_DATASET, "actives.sdf")
    INACTIVES = os.path.join(FULL_DATASET, "inactives.sdf")
    col = CollectionFromSDF(ACTIVES, INACTIVES)
    with pytest.raises(NotImplementedError):
        col.cluster(cutoff=0.3, fp="a")


# # BROKEN TESTS
# def test_to_dataframe():
#     col.add_features(Features.mordred, Features.topo)
#     df = col.to_dataframe()
#     assert df.shape == (25, 3390)
#     df = col.to_dataframe(Features.mordred)
#     assert df.shape == (18, 1240)


# # INTEGRATION TEST
# def test_creation_to_dataframe():
#     ACTIVES = os.path.join(FULL_DATASET, "actives.sdf")
#     INACTIVES = os.path.join(FULL_DATASET, "inactives.sdf")
#     GLIDE = os.path.join(FULL_DATASET, "glide_features.csv")
    
#     col = CollectionFromSDF(ACTIVES, INACTIVES)
#     col.add_features(Features.glide, Features.mordred, Features.topo, glide_csv=GLIDE)
#     df = col.to_dataframe()
#     df.to_csv("glide_mordred_topo.csv")
#     # model_df = pd.read_csv(os.path.join(DATA, "glide_mordred_topo.csv"), index_col=0)
#     # assert_frame_equal(df, model_df, check_dtype=False)
