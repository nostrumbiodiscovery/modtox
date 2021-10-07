from abc import ABC, abstractmethod
from modtox.modtox.new_models_classes.ML.dataset import DataSet
import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.feature_selection import SelectPercentile, chi2

class FeaturesSelector(ABC):
    def __init__(self, dataset: DataSet) -> None:
        super().__init__()
        self.dataset = dataset

    @abstractmethod
    def select(self):
        """Selects features and modifies DataSet."""

class ChiPercentileSelector(FeaturesSelector):
    def __init__(self, dataset: DataSet, percentile=50) -> None:
        super().__init__(dataset)
        self.percentile = percentile

    def select(self):
        """Select best features and modify dataframe and returns
        number of features"""
        X_abs = np.abs(self.dataset.X)
        selector = SelectPercentile(chi2, percentile=self.percentile)
        X_new = selector.fit_transform(X_abs, self.dataset.y)
        
        self.support = selector.get_support()  # T/F array -> T: feature kept, F: feature dropped
        columns_kept = [col for col, tf in zip(X_abs.columns, self.support) if tf == True]  # Columns 
        new_df = pd.DataFrame(X_new, columns=columns_kept)
        # Append two columns to maintain same structure as original dataframe
        new_df[self.dataset.y_col_name] = self.dataset.activity
        new_df[self.dataset.external_col_name] = self.dataset.is_external
        self.dataset.df = new_df
        return self.dataset.df.shape[1] - 2  # -2 for column y and ext_val

