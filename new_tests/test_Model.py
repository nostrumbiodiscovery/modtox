from pandas.core.frame import DataFrame
from sklearn.datasets import load_breast_cancer
import random
from pytest import approx

from modtox.modtox.new_models_classes.ML.dataset import DataSet
from modtox.modtox.new_models_classes.ML.model import Model
from modtox.modtox.new_models_classes.ML.selector import ChiPercentileSelector
from modtox.modtox.new_models_classes.ML.tuning import RandomHalvingSearch

data = load_breast_cancer(as_frame=True)
df = data["data"]
df["y"] = [bool(row) for row in data["target"]]
l = [True] * 114 + [False] * 455
random.shuffle(l)
df["is_external"] = l

# INTEGRATION TEST
def test_Model():
    m = Model(DataSet(df, y_col_name="y"))
    m.select_features(ChiPercentileSelector)
    m.train(RandomHalvingSearch)
    m.external_set_validation()
    assert m.summary["initial_features"] == 30
    assert m.summary["selected_features"] == 15
    assert m.summary["training_score"] == approx(0.98, abs=0.1)
    assert m.summary["accuracy_ext_val"] == approx(0.95, abs=0.1)
    assert m.summary["precision_ext_val"] == approx(0.95, abs=0.1)




    
