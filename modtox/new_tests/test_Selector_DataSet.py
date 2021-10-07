from modtox.modtox.new_models_classes.ML.dataset import DataSet
from modtox.modtox.new_models_classes.ML.selector import ChiPercentileSelector
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import random
from pytest import approx

data = load_breast_cancer(as_frame=True)
df = data["data"]
df["y"] = [bool(row) for row in data["target"]]
l = [True] * 114 + [False] * 455  # 20% external set
random.shuffle(l)
df["is_external"] = l

def random_df_gen():
    """Generates random 10x5 dataframe with random
    values, including NaN."""
    cols = ["a", "b", "c", "Activity", "is_external"]
    df = pd.DataFrame(columns=cols)
    for i in range(5):
        a = np.random.randn(10)
        mask = np.random.choice([1, 0], a.shape, p=[.2, .8]).astype(bool)
        a[mask] = np.nan
        df[cols[i]] = a 
    return df

def test_impute_dataset():
    df = random_df_gen()
    orig_df = df.copy()
    imp_df = DataSet(df).df    # Init calls impute
    assert orig_df.shape == imp_df.shape
    assert list(orig_df.columns) == list(imp_df.columns)
    for orig_row, imp_row in zip(orig_df.values, imp_df.values):
        for orig, imp in zip(orig_row, imp_row):
            if np.isnan(orig):
                assert imp == 0
            else:
                assert approx(imp) == orig  # approx because imputation truncates at 9 decimals

def test_dataset_properties():
    ds = DataSet(df, y_col_name="y")
    assert ds.df.shape == (569, 32)
    assert ds.X.shape == (569, 30)
    assert ds.X_train.shape == (455, 30)
    assert ds.X_ext.shape == (114, 30)
    assert ds.y_train.shape == (455,)
    assert ds.y_ext.shape == (114,)

def test_ChiPercentileSelector_from_selector():
    ds = DataSet(df, y_col_name="y")
    orig_cols = ds.df.columns
    orig_df_shape = ds.df.shape
    selector = ChiPercentileSelector(ds)
    n_new_fts = selector.select()
    
    model_kept_cols = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean concavity', 'radius error', 'perimeter error', 
        'area error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst compactness', 'worst concavity', 
        'worst concave points', ds.y_col_name, ds.external_col_name]
        
    kept_cols = [col for col, i in zip(orig_cols, selector.support) if i == True]
    assert model_kept_cols == kept_cols + [ds.y_col_name, ds.external_col_name]
    
    assert approx(orig_df_shape[1]/2, abs=1) == n_new_fts  # approx because of division remainder
    assert orig_df_shape[0] == ds.df.shape[0]

def test_ChiPercentileSelector_from_dataset():
    ds = DataSet(df, y_col_name="y")
    orig_cols = ds.df.columns
    orig_df_shape = ds.df.shape
    selector = ChiPercentileSelector(ds)
    n_new_fts = ds.select_features(selector)
    
    model_kept_cols = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean concavity', 'radius error', 'perimeter error', 
        'area error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst compactness', 'worst concavity', 
        'worst concave points', ds.y_col_name, ds.external_col_name]
        
    kept_cols = [col for col, i in zip(orig_cols, selector.support) if i == True]
    assert model_kept_cols == kept_cols + [ds.y_col_name, ds.external_col_name]
    
    assert approx(orig_df_shape[1]/2, abs=1) == n_new_fts  # approx because of division remainder
    assert orig_df_shape[0] == ds.df.shape[0]