from abc import ABC, abstractmethod
import pandas as pd

from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile, chi2


class FeaturesSelector(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def select(self):
        """Selects features and returns the X_selected
        dataframe."""

    def get_selected_columns(self, selector, original_columns):
        support = selector.get_support()
        columns_kept = [
            col for col, tf in zip(original_columns, support) if tf == True
        ]
        return columns_kept

class ChiPercentileSelector(FeaturesSelector):
    def __init__(self, percentile=50) -> None:
        self.percentile = percentile

    def select(self, X, y):
        """Select x% best features. Modifies the df from DataSet."""
        selector = SelectPercentile(chi2, percentile=self.percentile)
        new_arr = selector.fit(X, y)
        columns_kept = self.get_selected_columns()
        X_new = pd.DataFrame(new_arr, columns=columns_kept)
        return X_new

class RFECrossVal(FeaturesSelector):
    def __init__(self, step) -> None:
        self.step = step

    def select(self, X, y):
        self.initial_features = len(X.columns)
        self.svc = SVC(kernel="linear")
        self.rfecv = RFECV(estimator=self.svc, step=self.step, scoring="accuracy")
        new_arr = self.rfecv.fit_transform(X, y)
        columns_kept = self.get_selected_columns(self.rfecv, X.columns)
        X_new = pd.DataFrame(new_arr, columns=columns_kept)
        return X_new
    
    def get_scores(self):
        f = self.initial_features
        feat_num = [f-(self.step*i) for i in range(len(self.rfecv.grid_scores_))]
        if feat_num[-1] < 1:
            feat_num[-1] = 1
        return feat_num, self.rfecv.grid_scores_