from collections import Counter
from modtox.modtox.new_models_classes.col import CollectionFromSDF, CollectionSummarizer
from modtox.modtox.new_models_classes.Features.feat_enum import Features

import os
import pandas as pd
from pandas._testing import assert_frame_equal

DATA = os.path.join(os.path.dirname(__file__), "data")
ACTIVES = os.path.join(DATA, "actives.sdf")
INACTIVES = os.path.join(DATA, "inactives.sdf")
GLIDE = os.path.join(DATA, "glide_features.csv")

ML_DATA = os.path.join(DATA, "ML_test")
col = CollectionFromSDF(ACTIVES, INACTIVES)
col.add_features(Features.glide, glide_csv=GLIDE)
mol_w_glide = col.molecules[0]
mol_wo_glide = col.molecules[5]

def test_read_sdf():
    cole = CollectionFromSDF(ACTIVES, INACTIVES)
    assert len(cole.actives) == 9
    assert len(cole.inactives) == 16
    assert len(cole.actives) + len(cole.inactives) == len(cole.molecules)


def test_balance():
    init_len = len(col.molecules)
    col.balance(seed=1)
    assert len(col.actives) == len(col.inactives)
    assert (len(col.actives) * 2 + len(col.waste)) == init_len


def test_remove_random_molecule():    
    inactives_wo_glide = [mol for mol in col.molecules if mol.has_glide == False]
    col.balance(seed=1)
    assert all(mol in inactives_wo_glide for mol in col.waste)

def test_extract_external_set():
    ext_set = col.extract_external_set(0.2, seed=1)
    ext_tags = [mol for mol in col.molecules if mol.is_external == True]
    ext_set_names = {mol.name for mol in ext_set}
    ext_tags_names = {mol.name for mol in ext_tags}
    assert len(ext_set) == 4
    assert ext_set_names == ext_tags_names

def test_to_dataframe():
    col.add_features(Features.mordred, Features.topo)
    df = col.to_dataframe()
    assert df.shape == (25, 3390)
    col.balance(seed=1)
    df = col.to_dataframe(Features.mordred)
    assert df.shape == (18, 1240)
    return

def test_calculate_similarity():
    ACTIVES = os.path.join(ML_DATA, "actives.sdf")
    INACTIVES = os.path.join(ML_DATA, "inactives.sdf")
    col = CollectionFromSDF(ACTIVES, INACTIVES)
    col.add_topo()
    col.calculate_similarities()
    out = CollectionSummarizer(col)
    out.plot_similarities(bins=20)
    return

def test_representative_scaffolds():
    ACTIVES = os.path.join(ML_DATA, "actives.sdf")
    INACTIVES = os.path.join(ML_DATA, "inactives.sdf")
    col = CollectionFromSDF(ACTIVES, INACTIVES)
    out = CollectionSummarizer(col)
    out.draw_most_representative_scaffolds(10)

def test_plot_scaffolds():
    ACTIVES = os.path.join(ML_DATA, "actives.sdf")
    INACTIVES = os.path.join(ML_DATA, "inactives.sdf")
    col = CollectionFromSDF(ACTIVES, INACTIVES)
    out = CollectionSummarizer(col)
    out.plot_scaffolds()

# INTEGRATION TEST
def test_everything():
    # ACTIVES = os.path.join(ML_DATA, "actives.sdf")
    # INACTIVES = os.path.join(ML_DATA, "inactives.sdf")
    # GLIDE = os.path.join(ML_DATA, "glide_features.csv")
    col = CollectionFromSDF(ACTIVES, INACTIVES)
    col.add_features(glide_csv=GLIDE)
    col.balance(seed=1)
    col.extract_external_set(0.2, seed=1)
    df = col.to_dataframe()
    # df.to_csv("glide_mordred_topo.csv")
    model_df = pd.read_csv(os.path.join(DATA, "glide_mordred_topo.csv"), index_col=0)
    assert_frame_equal(df, model_df, check_dtype=False)
