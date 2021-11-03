from typing import Dict
import pandas as pd
from modtox.modtox.ML.tuning import (
    GridHalvingSearch,
    HyperparameterTuner,
    RandomHalvingSearch,
    RandomSearch,
)
from modtox.modtox.ML.dataset import DataSet
from modtox.modtox.ML.selector import FeaturesSelector, _RFECV, _PCA
from modtox.modtox.utils._custom_errors import MethodError

import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    matthews_corrcoef,
)
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Score:
    accuracy: float
    recall: float
    precision: float
    matthews: float
    f1: float
    conf_matrix: np


class ModelSummary:
    def __init__(self, selector, tuner, estimator, y_pred) -> None:
        self.selector = selector
        self.tuner = tuner
        self.estimator = estimator
        self.y_pred = y_pred

    def plot_feature_selection(self):
        x, y = self.selector.get_scores()
        xmax = x[np.argmax(y)]
        ymax = y.max()
        text = f"Accuracy: {ymax:.3f} at {xmax} features "

        plt.figure()
        plt.xlabel("Number of features selected")
        plt.xlim(0, x[0])
        plt.ylim(top=1.2)
        plt.ylabel("Cross validation score (accuracy)")
        plt.plot(x, y)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(
            arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60"
        )
        kw = dict(
            xycoords="data",
            textcoords="axes fraction",
            arrowprops=arrowprops,
            bbox=bbox_props,
            ha="right",
            va="top",
        )
        plt.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)
        plt.grid()
        plt.show()


class Model:
    dataset: DataSet
    summary: Dict

    def __init__(self, X_train, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train

    def reduce_dimensionality(self, selector_class, **kwargs):
        if selector_class is None:
            self.X_train_reduced = self.X_train
            return None

        selector = selector_class(**kwargs)
        self.fitted_selector = selector_class.fit_selector(self.X_train)
        self.X_train_reduced = self.fitted_selector.transform(self.X_train)
        return selector

    def train(self, tuner_class: HyperparameterTuner, **kwargs):
        tuner = tuner_class(self.X_train_reduced, self.y_train, **kwargs)
        self.best_votclf, train_scores = tuner.search(**kwargs)
        self.best_estimators = [
            tup[1] for tup in self.best_votclf.estimators
        ] + [self.best_votclf]
        return tuner, train_scores

    def predict_all_clfs(self, X_test, y_true):
        y_preds = {
            clf.__class__.__name__: clf.predict(X_test)
            for clf in self.best_estimators
        }
        scores = {
            clf_name: self.score(y_true, y_pred)
            for clf_name, y_pred in y_preds.items()
        }
        return scores

    @staticmethod
    def score(y_true, y_pred):
        return Score(
            accuracy_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            matthews_corrcoef(y_true, y_pred),
            f1_score(y_true, y_pred),
            confusion_matrix(y_true, y_pred),
        )

    def build_model(
        self, dimred_method, tuning_method, tests: Dict, **kwargs  # tests -> {test_name -> (X_test, y_test)}
    ):
        dimred_methods = {"pca": _PCA, "rfecv": _RFECV, "none": None}
        tuning_methods = {
            "randomsearch": RandomSearch,
            "randomhalving": RandomHalvingSearch,
            "gridhalving": GridHalvingSearch,
        }
        choosen_dimred_method = self._choose_method(
            dimred_method, dimred_methods, "dimensionality reduction"
        )
        choosen_tuning_method = self._choose_method(
            tuning_method, tuning_methods, "hyperparameter tuning"
        )

        selector = self.reduce_dimensionality(choosen_dimred_method, **kwargs)
        tuner, train_scores = self.train(choosen_tuning_method, **kwargs)

        test_scores = {
            set_name: self.predict_all_clfs(X_test, y_test)
            for set_name, (X_test, y_test) in tests.items()
        }

        return train_scores, test_scores, self.best_estimators

    @staticmethod
    def _choose_method(method: str, methods: Dict, technique: str):
        if method not in methods:
            raise MethodError(
                f"Selected method {method!r} is not in implemented for {technique} "
                f"Implemented methods are: {', '.join(methods.keys())}"
            )
        chosen_method = methods[method]
        return chosen_method
