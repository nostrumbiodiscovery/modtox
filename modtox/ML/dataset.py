import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from modtox.modtox.utils._custom_errors import BalancingError, ScalingError

class DataSet:
    """Mapping is done via dataframe index, so all steps must maintain it."""
    def __init__(self, df: pd.DataFrame, test_prop=0.2, y_col_name="Activity") -> None:
        self.original = df
        self.test_prop = test_prop
        self.y_col_name = y_col_name
        
    def transform(self, outliers_threshold, resampling_method, scaling_method):
        strat_train_set, strat_test_set = self._stratified_split(self.original)
        X_train, y_train, discarded, fitted_scaler = self._process_train(strat_train_set, outliers_threshold, resampling_method, scaling_method)
        
        self._drop_columns(strat_test_set, ["cluster_id", "cluster_members", "is_centroid"])
        X_test, y_test = self._clean_to_Xy(strat_test_set, fitted_scaler)
        
        X_test_reg, y_test_reg, X_test_out, y_test_out = self._split_test(strat_test_set, discarded, outliers_threshold, fitted_scaler)

        sets = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_test_reg": X_test_reg,
            "y_test_reg": y_test_reg,
            "X_test_out": X_test_out,
            "y_test_out": y_test_out,
        }
        return sets
    
    def _split_test(self, strat_test, discarded, outliers_threshold, fitted_scaler):
        test = strat_test.append(discarded)
        test_reg = test[test["cluster_members"] >= outliers_threshold]
        test_out = test[test["cluster_members"] < outliers_threshold]
        
        self._drop_columns(test_reg, ["cluster_id", "cluster_members", "is_centroid"])
        self._drop_columns(test_out, ["cluster_id", "cluster_members", "is_centroid"])
        
        X_test_reg, y_test_reg = self._clean_to_Xy(test_reg, fitted_scaler)
        X_test_out, y_test_out = self._clean_to_Xy(test_out, fitted_scaler)
        
        return X_test_reg, y_test_reg, X_test_out, y_test_out

    def _process_train(self, strat_train, outliers_threshold, resampling_method, scaling_method):
        train_clean, outliers = self._remove_outliers(strat_train, n_members_threshold=outliers_threshold)
        train_resampled, waste = self._resample(train_clean, method=resampling_method)
        discarded = outliers.append(waste)
        self._drop_columns(train_resampled, ["cluster_id", "cluster_members", "is_centroid"])
        fitted_scaler = self._fit_scaler(train_resampled, method=scaling_method)
        X_train, y_train = self._clean_to_Xy(train_resampled, fitted_scaler)
        return X_train, y_train, discarded, fitted_scaler

    def _stratified_split(self, df) -> pd.DataFrame:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.test_prop, random_state=42)
        for train_index, test_index in splitter.split(df, df[self.y_col_name]):
            strat_train_set = df.loc[train_index]
            strat_test_set = df.loc[test_index]
        return strat_train_set, strat_test_set

    def _remove_outliers(self, df, n_members_threshold: int = 10, col_name="cluster_members"):
        new_df = df[df[col_name] >= n_members_threshold]
        outliers = df[df[col_name] < n_members_threshold]
        return new_df, outliers

    def _resample(self, df, method: str = "oversampling"):
        methods = {
            "oversampling": RandomOverSampler,
            "undersampling": RandomUnderSampler,
            "none": None
        }

        if method not in methods.keys():
            raise BalancingError(
                f"Selected method {method!r} is not in implemented balancing "
                f"methods: {', '.join(methods.keys())}"
            )
        chosen_method = methods[method]
        if chosen_method is None:
            waste = pd.DataFrame()
            return df, waste

        X = df.drop([self.y_col_name], axis=1)
        y = df[self.y_col_name]
        
        ros = methods[method](random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        X_resampled.insert(0, self.y_col_name, y_resampled)
        
        resampled = X_resampled
        resampled.index = ros.sample_indices_
        
        train_clean_indices = set(df.index)
        train_resampled_indices = set(resampled.index)
        waste = df.loc[train_clean_indices.difference(train_resampled_indices)]
        
        return resampled, waste
    
    @staticmethod
    def _impute(X):
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        X_imp.index = X.index
        return X_imp
    
    def _fit_scaler(self, train_set, method="standarize"):
        methods = {
            "standarize": StandardScaler,
            "normalize": MinMaxScaler,
            "none": None
        }

        if method not in methods:
            raise ScalingError(
                f"Selected method {method!r} is not in implemented scaling "
                f"methods: {', '.join(methods.keys())}"
            )
        
        choosen_scaler = methods[method]
        
        if choosen_scaler is None:
            return None
        else:
            X_train = train_set.drop(self.y_col_name, axis=1)
            X_train_imp = self._impute(X_train) 
            scaler = methods[method]()
            fitted_scaler = scaler.fit(X_train_imp)
            return fitted_scaler

    @staticmethod
    def _scale(X, fitted_scaler):
        if fitted_scaler is None:
            return X
        else:
            X_sc = pd.DataFrame(fitted_scaler.transform(X), columns=X.columns)
            X_sc.index = X.index
            return X_sc


    def _clean_to_Xy(self, df, fitted_scaler):
        X = df.drop([self.y_col_name], axis=1)
        X_imp = self._impute(X)
        X_sc = self._scale(X_imp, fitted_scaler)
        
        y = df[self.y_col_name]
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        return X_sc, y_enc
        
    @staticmethod
    def _drop_columns(df, cols_to_remove):
        for column in cols_to_remove:
            if column in list(df.columns):
                df.drop([column], axis=1, inplace=True)

    