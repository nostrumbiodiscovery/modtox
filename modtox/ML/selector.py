from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

class FeaturesSelector(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit_selector(self):
        """Selects features and returns fitted selector.
        Must accept X and y even if only uses X."""

    @abstractmethod
    def plotting_data(self):
        """Returns data for performance analysis."""


class _PCA(FeaturesSelector):
    def __init__(self, variance=0.95) -> None:
        """
        Parameters
        ----------
        variance : float, optional
            Desired explained variance to keep, by default 0.95
        """
        self.variance = variance

    def fit_selector(self, X, y=None):
        """Fits selector.

        Parameters
        ----------
        X : pd.DataFrame
            X_test
        y : None, optional
            Just to keep same format to other selectors, by default None

        Returns
        -------
        Fitted sklearn.Selector
            Fitted selector for transorming test and/or train features.
        """
        self.X = X
        self.pca = PCA(n_components=self.variance)
        fitted_selector = self.pca.fit(X)
        return fitted_selector

    def plotting_data(self):
        """Returns the explained variance against the dimensions 
        for plotting.

        Returns
        -------
        List[int], np.array[float]
            x, y for plotting.
        """
        pca = PCA()
        pca.fit(self.X)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        return range(0, len(cumsum)), cumsum


class _RFECV(FeaturesSelector):
    def __init__(self, step) -> None:
        self.step = step

    def fit_selector(self, X, y):
        """Fits selector.

        Parameters
        ----------
        X : pd.DataFrame
            X_train
        y : np.array or pd.DataFrame
            y_train, labels

        Returns
        -------
        Fitted sklearn.selector
            Fitted selector to train set.
        """
        self.svc = SVC(kernel="linear")
        self.rfecv = RFECV(
            estimator=self.svc, 
            step=self.step,
            cv=2,
            min_features_to_select=1
        )
        fitted_selector = self.rfecv.fit(X, y)
        return fitted_selector

    def plotting_data(self):
        """To plot number of selected features against accuracy

        Returns
        -------
        List[int], List[float]
            x, y for plotting
        """
        n_init_feat = len(self.rfecv.get_support())
        feat_num = list(range(1, n_init_feat, self.step)) + [n_init_feat] # Not totally accurate. For large number of features the error is neglible.
        return feat_num, self.rfecv.grid_scores_

