from typing import Dict, List
import pandas as pd
from modtox.modtox.ML.tuning import HyperparameterTuner, RandomHalvingSearch
from modtox.modtox.ML.dataset import DataSet
from modtox.modtox.ML.selector import FeaturesSelector, RFECrossVal
from dataclasses import dataclass
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt

@dataclass
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
        text= f"Accuracy: {ymax:.3f} at {xmax} features "

        plt.figure()
        plt.xlabel("Number of features selected")
        plt.xlim(0, x[0])
        plt.ylim(top=1.2)
        plt.ylabel("Cross validation score (accuracy)")
        plt.plot(x, y)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        plt.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
        plt.grid()
        plt.show()

    def get_estimators_accuracy(self, X_test, y_test):
        estimator_list = [tup[1] for tup in self.estimator.estimators] + [self.estimator]
        records = list()
        for clf in estimator_list:
            rec = {"classifier": clf.__class__.__name__}
            y_pred = clf.predict(X_test)
            
            rec["accuracy"] = accuracy_score(y_test, y_pred)
            rec["matrix"] = confusion_matrix(y_test, y_pred)
            rec["precision"] = precision_score(y_test, y_pred)
            rec["recall"] = recall_score(y_test, y_pred)
            rec["f1"] = f1_score(y_test, y_pred)
            records.append(rec)
        
        df = pd.DataFrame(records)
        return df
       
class Model:
    """Represents a model to predict new molecules.
    1 to 1 relationship with: DataSet, FeatureSelector, HyperTuner.
    Many to 1 relationship with Collection. """

    dataset: DataSet
    summary: Dict

    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def select_by_RFECV(self, step=10):
        selector = RFECrossVal(step=step)
        self.X_selected = selector.select(self.X_train, self.y_train)
        return selector

    def train(self, tuner: HyperparameterTuner):
        tuner = tuner(self.X_train, self.y_train)
        self.estimator = tuner.search()
        return tuner

    def test(self):
        y_pred = self.estimator.predict(self.X_test)
        return y_pred

    def build_model(self, step=10) -> ModelSummary:
        selector = self.select_by_RFECV(step=step)
        tuner = self.train(RandomHalvingSearch)
        y_pred = self.test()
        return ModelSummary(selector, tuner, self.estimator, y_pred)

