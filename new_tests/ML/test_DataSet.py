from modtox.modtox.ML.dataset import DataSet
from modtox.modtox.utils._custom_errors import BalancingError, ScalingError
import pandas as pd

import os

import pytest

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REDUCED_DATASET = os.path.join(DATA, "reduced_dataset")
FULL_DATASET = os.path.join(DATA, "full_dataset")

ACTIVES = os.path.join(REDUCED_DATASET, "actives.sdf")
INACTIVES = os.path.join(REDUCED_DATASET, "inactives.sdf")
GLIDE_CSV = os.path.join(REDUCED_DATASET, "glide_features.csv")

OUTPUT_FOLDER = os.path.join(DATA, "tests_outputs")

df = pd.read_csv(os.path.join(FULL_DATASET, "glide_mordred_topo.csv"), index_col=0)

def test_remove_outliers():
    ds = DataSet(df)
    new_df, outliers = ds.remove_outliers(ds.original_ds, n_members_threshold=10, col_name="cluster_members")
    assert all(x >= 10 for x in list(new_df["cluster_members"]))
    assert all(x < 10 for x in list(outliers["cluster_members"]))
    assert new_df.shape[0] == 489
    assert outliers.shape[0] == 206

def test_stratified_split():
    ds = DataSet(df)
    original = ds.original_ds["Activity"].value_counts() / len(ds.original_ds)
    strat_train_set, strat_test_set = ds.stratified_split(ds.original_ds)
    train = strat_train_set["Activity"].value_counts() / len(strat_train_set)
    test = strat_test_set["Activity"].value_counts() / len(strat_test_set)
    assert list(original) == [0.6532374100719425, 0.34676258992805753]
    assert list(train) == [0.6528776978417267, 0.3471223021582734]
    assert list(test) == [0.6546762589928058, 0.34532374100719426]

def test_preprocess_standarize():
    ds = DataSet(df)
    X, y = ds.preprocess(ds.original_ds, scaling="standarize")
    assert X.shape == (695, 3829)
    assert y.shape == (695,)

def test_preprocess_normalize():
    ds = DataSet(df)
    X, y = ds.preprocess(ds.original_ds, scaling="normalize")
    assert X.shape == (695, 3829)
    assert y.shape == (695,)
    assert all(0 <= x <= 1 for x in list(X["Score"]))

def test_preprocess():
    ds = DataSet(df)
    with pytest.raises(ScalingError):
        X, y = ds.preprocess(ds.original_ds, scaling="randomstring")
    
def test_balance_oversampling():
    ds = DataSet(df)
    resampled_df = ds.balance(ds.original_ds, method="oversampling")
    assert resampled_df.shape == (908, 3833)

def test_balance_undersampling():
    ds = DataSet(df)
    resampled_df = ds.balance(ds.original_ds, method="undersampling")
    assert resampled_df.shape == (482, 3833)

def test_balance_error():
    ds = DataSet(df)
    with pytest.raises(BalancingError):
        resampled_df = ds.balance(ds.original_ds, method="randomstring")

# INTEGRATION TEST
def test_prepare():
    ds = DataSet(df)
    X_train, y_train, X_test, y_test = ds.prepare(
        outliers_threshold=10, 
        balancing_method="oversampling", 
        scaling_method="normalize"
    )
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

