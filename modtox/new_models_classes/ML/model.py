from typing import Dict
from collections import Counter

from modtox.modtox.new_models_classes.ML.tuning import HyperparameterTuner
from modtox.modtox.new_models_classes.ML.dataset import DataSet
from modtox.modtox.new_models_classes.ML.selector import FeaturesSelector

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

# Tuner and Selector could be defined in init.
class Model:
    """Represents a model to predict new molecules.
    1 to 1 relationship with: DataSet, FeatureSelector, HyperTuner.
    Many to 1 relationship with Collection. """
    
    dataset: DataSet
    summary: Dict
    
    def __init__(self, dataset: DataSet) -> None:
        self.dataset = dataset

        self.summary = dict()
        self.summary["initial_features"] = self.dataset.df.shape[1] - 2

    def select_features(self, selector: FeaturesSelector):
        """Calls selector and modifies DataSet."""
        selector = selector(self.dataset)
        n_features = self.dataset.select_features(selector)
        self.summary["selected_features"] = n_features

    def train(self, tuner: HyperparameterTuner):
        """Decides best parameter combination."""
        tuner = tuner(self.dataset.X_train, self.dataset.y_train)
        self.estimator = tuner.search()
        self.summary["training_score"] = tuner.score
        self.summary["best_parameters"] = tuner.best_params

    def external_set_validation(self):
        """Tests model agains external validation set"""
        y_pred = self.estimator.predict(self.dataset.X_ext)
        tn, fp, fn, tp = confusion_matrix(self.dataset.y_ext, y_pred).ravel()
        conf_matrix = f"tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}"
        accuracy = accuracy_score(self.dataset.y_ext, y_pred)
        precision = precision_score(self.dataset.y_ext, y_pred)
        self.summary["confusion_matrix_ext_val"] = conf_matrix
        self.summary["accuracy_ext_val"] = accuracy
        self.summary["precision_ext_val"] = precision