import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import itertools

class BalancingError(Exception):
    pass

class ScalingError(Exception):
    pass

class DataSet:
    def __init__(self, df: pd.DataFrame, test_prop=0.2, y_col_name="Activity") -> None:
        self.original_ds = df
        self.test_prop = test_prop
        self.y_col_name = y_col_name
    
    def remove_outliers(self, df, n_members_threshold: int = 10, col_name="cluster_members"):
        new_df = df[df[col_name] >= n_members_threshold]
        outliers = df[df[col_name] < n_members_threshold]
        return new_df, outliers

    def balance(self, df, method: str = "oversampling"):
        methods = {
            "oversampling": RandomOverSampler,
            "undersampling": RandomUnderSampler
        }
        if method not in methods.keys():
            raise BalancingError(
                f"Selected method {method!r} is not in implemented balancing "
                f"methods: {', '.join(methods.keys())}"
            )
        X = df.drop([self.y_col_name], axis=1)
        y = df[self.y_col_name]
        ros = methods[method](random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        X_resampled[self.y_col_name] = y_resampled
        return X_resampled

    def stratified_split(self, df):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.test_prop, random_state=42)
        for train_index, test_index in splitter.split(df, df[self.y_col_name]):
            strat_train_set = df.loc[train_index]
            strat_test_set = df.loc[test_index]
        return strat_train_set, strat_test_set
    
    def preprocess(self, df, scaling="standarize"):
        scaler = {
            "standarize": StandardScaler,
            "normalize": MinMaxScaler,
        }

        if scaling not in scaler:
            raise ScalingError(
                f"Selected method {scaling!r} is not in implemented scaling "
                f"methods: {', '.join(scaler.keys())}"
            )

        X = df.drop([self.y_col_name], axis=1)
        y = df[self.y_col_name]
        
        cols_to_remove = ["cluster_id", "cluster_members", "is_centroid"]
        
        for column in cols_to_remove:
            if column in list(X.columns):
                X.drop([column], axis=1, inplace=True)
        
        pipeline = Pipeline([
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)),
            ("scaler", scaler[scaling]()),  # Check also normalization: MinMaxScaler()
        ])
        
        X_prepared = pd.DataFrame(pipeline.fit_transform(X), columns=X.columns)
        y_prepared = LabelEncoder().fit_transform(y)
        
        return X_prepared, y_prepared

    def prepare(self, outliers_threshold, balancing_method, scaling_method):
        strat_train_set, strat_test_set = self.stratified_split(self.original_ds)
        X_test, y_test = self.preprocess(strat_test_set, scaling=scaling_method)

        train_clean_ds, outliers = self.remove_outliers(strat_train_set, n_members_threshold=outliers_threshold)
        train_resampled = self.balance(train_clean_ds, method=balancing_method)
        X_train, y_train = self.preprocess(train_resampled, scaling=scaling_method)
        return X_train, y_train, X_test, y_test