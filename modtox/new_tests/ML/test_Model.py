from modtox.modtox.ML.model import Model, ModelSummary
from modtox.modtox.ML.tuning import RandomHalvingSearch
from modtox.modtox.utils._custom_errors import MethodError
from modtox.modtox.ML.selector import FeaturesSelector, _RFECV, _PCA

from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np
import os
import pandas as pd

from unittest.mock import MagicMock, patch, call
import pytest

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN = os.path.join(DATA, "tests_outputs", "train_set.csv")
TEST = os.path.join(DATA, "tests_outputs", "test_set.csv")
TEST_REG = os.path.join(DATA, "tests_outputs", "testreg_set.csv")
TEST_OUT = os.path.join(DATA, "tests_outputs", "testout_set.csv")


def to_Xy(csv):
    df = pd.read_csv(csv, index_col=0)
    X = df.drop("Activity", axis=1)
    y = df["Activity"].to_numpy()
    return X, y


X_train, y_train = to_Xy(TRAIN)
X_test, y_test = to_Xy(TEST)
X_testreg, y_testreg = to_Xy(TEST_REG)
X_testout, y_testout = to_Xy(TEST_OUT)


m = Model(X_train, y_train)


def test_reduce_dimensionality_none():
    m.reduce_dimensionality(None)
    pd.testing.assert_frame_equal(m.X_train, m.X_train_reduced)


@patch("modtox.modtox.ML.model._PCA.fit_selector")
@patch("modtox.modtox.ML.model._PCA.__init__", return_value=None)
def test_reduce_dimensionality_PCA(init, fit):
    m.reduce_dimensionality(_PCA, variance=0.95, other="kwargs")
    init.assert_called_once_with(variance=0.95, other="kwargs")
    fit.assert_called_once_with(m.X_train)


lr = LogisticRegression()
knn = KNeighborsClassifier()
vot_clf = VotingClassifier(estimators=[("lr", lr), ("knn", knn)])


@patch(
    "modtox.modtox.ML.model.RandomHalvingSearch.search", return_value=vot_clf
)
@patch(
    "modtox.modtox.ML.model.RandomHalvingSearch.__init__", return_value=None
)
def test_train(init, search):
    X_train_mock = MagicMock()
    y_train_mock = MagicMock()
    m.X_train_reduced = X_train_mock
    m.y_train = y_train_mock
    m.train(RandomHalvingSearch, some="kwarg", other="kwargs")
    init.assert_called_once_with(
        X_train_mock, y_train_mock, some="kwarg", other="kwargs"
    )
    search.assert_called_once_with(some="kwarg", other="kwargs")
    assert set(m.best_estimators) == {lr, knn, vot_clf}


@patch.object(LogisticRegression, "predict", return_value="lr")
@patch.object(KNeighborsClassifier, "predict", return_value="knn")
@patch.object(VotingClassifier, "predict", return_value="votclf")
def test_predict_all(votclf_predict, knn_predict, lr_predict):
    lr = LogisticRegression()
    knn = KNeighborsClassifier()
    vot_clf = VotingClassifier(estimators=[("lr", lr), ("knn", knn)])
    m.best_estimators = [lr, knn, vot_clf]

    y_preds = m.predict_all_clfs("X_test")
    votclf_predict.assert_called_once_with("X_test")
    knn_predict.assert_called_once_with("X_test")
    lr_predict.assert_called_once_with("X_test")
    assert y_preds["LogisticRegression"] == "lr"
    assert y_preds["KNeighborsClassifier"] == "knn"
    assert y_preds["VotingClassifier"] == "votclf"


@patch.object(Model, "reduce_dimensionality")
@patch.object(Model, "train")
@patch.object(Model, "predict_all_clfs")
def test_build_model(pred, train, red):
    m.best_estimators = ["a", "b"]
    y_preds, estims = m.build_model(
        "pca",
        "randomhalving",
        {"s1": "set1", "s2": "set2"},
        variance=0.95,
        voting="hard",
    )
    red.assert_called_once_with(_PCA, variance=0.95, voting="hard")
    train.assert_called_once_with(
        RandomHalvingSearch, variance=0.95, voting="hard"
    )
    assert list(pred.call_args_list) == [call("set1"), call("set2")]


def test_choose_method():
    methods = {"m1": str, "m2": int, "m3": None}
    assert m._choose_method("m1", methods, "placeholder") == str
    assert m._choose_method("m2", methods, "placeholder") == int
    assert m._choose_method("m3", methods, "placeholder") is None
    with pytest.raises(MethodError):
        assert m._choose_method("randomstring", methods, "placeholder") is None


"""{'KNeighborsClassifier': 0.8057553956834532, 'SVC': 0.9280575539568345, 'LogisticRegression': 0.9136690647482014, 'DecisionTreeClassifier': 0.9712230215827338, 'BernoulliNB': 0.8633093525179856, 'VotingClassifier': 0.9424460431654677}
"""
