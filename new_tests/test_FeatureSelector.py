from mock.mock import PropertyMock
from modtox.modtox.new_models_classes.ML.selector import ChiPercentileSelector, RFECrossVal, FeaturesSelector
from modtox.modtox.new_models_classes.ML.dataset import DataSet
from sklearn.feature_selection import RFECV
import pandas as pd
import os
from mock import patch

DATA = os.path.join(os.path.dirname(__file__), "data")
ML_DATA = os.path.join(DATA, "ML_test")

df = pd.read_csv(os.path.join(ML_DATA, "glide_mordred_topo.csv"), index_col=0)
ds = DataSet(df)
X, y = ds.preprocess(ds.original_ds, scaling="normalize")

def test_get_selected_columns():
    class SelectorMock():
        def get_support(self):
            return [True, False, True]
    
    selector = SelectorMock()
    columns = ["A", "B", "C"]
    fs = ChiPercentileSelector()
    columns_kept = fs.get_selected_columns(selector, columns)
    assert columns_kept == ["A", "C"]
    return

def test_rfecv_select():
    selector = RFECrossVal(step=1000)
    X_new = selector.select(X, y)
    for column in X_new.columns:
        assert list(X[column]) == list(X_new[column])
    return

def test_get_scores_rfecv():
    selector = RFECrossVal(step=100)
    X_new = selector.select(X, y)
    feat_num, scores = selector.get_scores()
    assert len(feat_num) == len(scores)
    assert feat_num[-1] == 1

