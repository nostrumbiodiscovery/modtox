from pandas.core.frame import DataFrame
from modtox.modtox.ML.dataset import DataSet
from modtox.modtox.utils._custom_errors import BalancingError, ScalingError

import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pytest
from unittest.mock import patch

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REDUCED_DATASET = os.path.join(DATA, "reduced_dataset")
FULL_DATASET = os.path.join(DATA, "full_dataset")

ACTIVES = os.path.join(REDUCED_DATASET, "actives.sdf")
INACTIVES = os.path.join(REDUCED_DATASET, "inactives.sdf")
GLIDE_CSV = os.path.join(REDUCED_DATASET, "glide_features.csv")

OUTPUT_FOLDER = os.path.join(DATA, "tests_outputs")

df = pd.read_csv(
    os.path.join(FULL_DATASET, "glide_mordred_topo.csv"), index_col=0
)


def test_stratified_split():
    ds = DataSet(df)
    original = ds.original["Activity"].value_counts() / len(ds.original)
    strat_train_set, strat_test_set = ds._stratified_split(ds.original)
    train = strat_train_set["Activity"].value_counts() / len(strat_train_set)
    test = strat_test_set["Activity"].value_counts() / len(strat_test_set)
    assert list(original) == [0.6532374100719425, 0.34676258992805753]
    assert list(train) == [0.6528776978417267, 0.3471223021582734]
    assert list(test) == [0.6546762589928058, 0.34532374100719426]

def test_remove_outliers():
    ds = DataSet(df)
    clean, outliers = ds._remove_outliers(
        ds.original, n_members_threshold=10, col_name="cluster_members"
    )
    assert all(x >= 10 for x in list(clean["cluster_members"]))
    assert all(x < 10 for x in list(outliers["cluster_members"]))
    assert clean.shape[0] == 489
    assert outliers.shape[0] == 206
    joined = clean.append(outliers)
    joined = joined.sort_index()
    pd.testing.assert_frame_equal(joined, df)

def test_resampling_oversampling():
    ds = DataSet(df)
    resampled, waste = ds._resample(ds.original, method="oversampling")
    assert resampled.shape == (908, 3833)
    assert waste.empty

def test_resampling_undersampling():
    ds = DataSet(df)
    resampled, waste = ds._resample(ds.original, method="undersampling")
    joined = resampled.append(waste)
    joined = joined.sort_index()
    pd.testing.assert_frame_equal(joined, df)
    assert resampled.shape == (482, 3833)
    assert waste.shape == (213, 3833)

def test_no_resampling():
    ds = DataSet(df)
    resampled, waste = ds._resample(ds.original, method="none")
    pd.testing.assert_frame_equal(resampled, df)
    assert waste.empty

def test_balance_error():
    ds = DataSet(df)
    with pytest.raises(BalancingError):
        resampled_df = ds._resample(ds.original, method="randomstring")


def test_impute():
    df = pd.DataFrame([[0, None, "string"], [None, 21, "string2"]], columns=["a", "b", "c"])
    df.index = ["in1", "in2"]
    
    ds = DataSet("placeholder")
    df_imp = ds._impute(df)
    
    model = pd.DataFrame([[0, 0, "string"], [0, 21, "string2"]], columns=["a", "b", "c"])
    model.index = ["in1", "in2"]
    pd.testing.assert_frame_equal(model, df_imp, check_dtype=False)

def test_fit_scaler_standarize():
    ds = DataSet(df)
    X = df.drop("Activity", axis=1)
    X.index = np.random.randint(1, size=X.shape[0])
    X = ds._impute(X)
    fitted_scaler = ds._fit_scaler(X, method="standarize")
    assert isinstance(fitted_scaler, StandardScaler)
    assert hasattr(fitted_scaler, "scale_")

def test_fit_scaler_normalize():
    ds = DataSet(df)
    X = df.drop("Activity", axis=1)
    X.index = np.random.randint(1, size=X.shape[0])
    X = ds._impute(X)
    fitted_scaler = ds._fit_scaler(X, method="normalize")
    assert isinstance(fitted_scaler, MinMaxScaler)
    assert hasattr(fitted_scaler, "scale_")

def test_fit_scaler_none():
    ds = DataSet(df)
    X = df.drop("Activity", axis=1)
    fitted_scaler = ds._fit_scaler(X, method="none")
    assert fitted_scaler is None

def test_scale_error():
    ds = DataSet(df)
    with pytest.raises(ScalingError):
        X_sc = ds._fit_scaler(df, method="randomstring")

def test_scale():
    ds = DataSet("placeholder")
    X = pd.DataFrame(np.random.randint(0,100,size=(50, 6)), columns=list('ABCDEF'))
    fitted_scaler = StandardScaler().fit(X)
    X_sc = ds._scale(X, fitted_scaler)
    assert all(x == pytest.approx(0, abs=0.01) for x in list(X_sc.mean()))
    assert X.shape == X_sc.shape
    assert list(X.columns) == list(X_sc.columns)
    assert list(X.index) == list(X_sc.index)

def test_scale_none():
    ds = DataSet("placeholder")
    X = pd.DataFrame(np.random.randint(0,100,size=(50, 6)), columns=list('ABCDEF'))
    fitted_scaler = None
    X_sc = ds._scale(X, fitted_scaler)
    pd.testing.assert_frame_equal(X_sc, X)  

def returnX(value, method=None):
    return value
@patch("modtox.modtox.ML.dataset.DataSet._impute", side_effect=returnX)
@patch("modtox.modtox.ML.dataset.DataSet._scale", side_effect=returnX)
def test_clean_to_Xy(scale, impute):
    ds = DataSet(df)
    X, y = ds._clean_to_Xy(df, "abc")
    scale.assert_called_once()
    impute.assert_called_once()
    assert "Activity" not in X.columns
    assert all(x == 0 or x == 1 for x in y)

def test_drop_columns():
    ds = DataSet("placeholder")
    df = pd.DataFrame([1, 2, 4, 5], columns=["numbers"])
    ds._drop_columns(df, ["numbers"])
    assert df.empty

def test_integration():
    ds = DataSet(df)
    d = ds.transform(10, "undersampling", "normalize")
    assert d["X_train"].shape[0] == d["y_train"].shape[0]
    assert d["X_test"].shape[0] == d["y_test"].shape[0]
    assert d["X_test_reg"].shape[0] == d["y_test_reg"].shape[0]
    assert d["X_test_out"].shape[0] == d["y_test_out"].shape[0]
