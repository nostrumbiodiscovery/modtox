from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import GridSearchCV


class FeaturesSelector(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit_selector(self):
        """Selects features and returns fitted selector."""

    @abstractmethod
    def plotting_data(self):
        """Returns data for performance analysis."""


class _PCA(FeaturesSelector):
    def __init__(self, variance=0.95) -> None:
        self.variance = variance

    def fit_selector(self, X, y=None):
        self.pca = PCA(n_components=self.variance)
        fitted_selector = self.pca.fit(X)
        return fitted_selector

    def plotting_data(self):
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        return range(0, len(cumsum)), cumsum


class _RFECV(FeaturesSelector):
    def __init__(self, step) -> None:
        self.step = step

    def fit_selector(self, X, y):
        self.svc = SVC(kernel="linear")
        self.rfecv = RFECV(
            estimator=self.svc, step=self.step, scoring="accuracy"
        )
        fitted_selector = self.rfecv.fit(X, y)
        return fitted_selector

    def plotting_data(self):
        n_init_feat = len(self.rfecv.get_support())
        feat_num = [
            n_init_feat - (self.step * i)
            for i in range(len(self.rfecv.grid_scores_))
        ]
        if feat_num[-1] < 1:
            feat_num[-1] = 1
        return feat_num, self.rfecv.grid_scores_


class kPCA:
    def __init__(self, variance=0.95) -> None:
        self.variance = variance

    def select(self):
        kpca = KernelPCA(n_components=self.variance)
        params = {
            "kernel": ["linear", "rbf", "sigmoid"],
            "gamma": np.linspace(0.03, 0.05, 10),
        }
