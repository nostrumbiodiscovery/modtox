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
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe resulting from the modtox.Collection.to_dataframe()
            Should only contain:
                - Cluster information (cluster_members column)
                - Features
                - y_col: Activity
        test_prop : float, optional
            Split proportion for test, by default 0.2
        y_col_name : str, optional
            Name of the column containing the labels, by default "Activity"
        """
        self.original = df
        self.test_prop = test_prop
        self.y_col_name = y_col_name
        
    def transform(self, outliers_threshold, resampling_method, scaling_method):
        """Wrapper function for dividing the given dataframe into 4 sets: train, test,
        test regulars and test outliers.

        Parameters
        ----------
        outliers_threshold : int
            Threshold of cluster members for which a molecule will be
            considered an outlier,
        resampling_method : str
            "undersampling", "oversampling" or "none".
        scaling_method : str
            "normalize", "standarize" or "none".

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing all the dataframes/labels for ML.
        """
        strat_train_set, strat_test_set = self._stratified_split(self.original)
        X_train, y_train, discarded, fitted_scaler = self._process_train(strat_train_set, outliers_threshold, resampling_method, scaling_method)
        X_test_reg, y_test_reg, X_test_out, y_test_out = self._split_test(strat_test_set, discarded, outliers_threshold, fitted_scaler)
        
        self._drop_columns(strat_test_set, ["cluster_id", "cluster_members", "is_centroid"])
        X_test, y_test = self._clean_to_Xy(strat_test_set, fitted_scaler)
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
        """Splits the test in different test sets, adding the discarded/outliers.
            1. Merges test set + outliers + "undersampled"
            2. Splits in regulars or outliers by outlier threshold.
            3. Drops cluster information. 
            4. Splits, checking for empty dataframes.
                If outlier threshold is 0, outlier test sets will be empty.
            Empty df are set to None. 

        Parameters
        ----------
        strat_test : pd.DataFrame
            test set (after stratified split)
        discarded : pd.DataFrame
            outliers + removed molecules if "undersampling".
        outliers_threshold : int
            Threshold of cluster members for which a molecule will be
            considered an outlier,
        fitted_scaler : sklearn.Scaler
            Scaler fitted to the train set

        Returns
        -------
        X_test_reg, y_test_reg : pd.DataFrame, np.array
            Regulars test set.
        Same for outliers.
        """
        test = strat_test.append(discarded)
        test_reg = test[test["cluster_members"] > outliers_threshold]
        test_out = test[test["cluster_members"] <= outliers_threshold]
        
        self._drop_columns(test_reg, ["cluster_id", "cluster_members", "is_centroid"])
        self._drop_columns(test_out, ["cluster_id", "cluster_members", "is_centroid"])
        
        if test_reg.empty:
            X_test_reg, y_test_reg = None, None
        else:
            X_test_reg, y_test_reg = self._clean_to_Xy(test_reg, fitted_scaler)
        
        if test_out.empty:
            X_test_out, y_test_out = None, None
        else:
            X_test_out, y_test_out = self._clean_to_Xy(test_out, fitted_scaler)
        
        return X_test_reg, y_test_reg, X_test_out, y_test_out

    def _process_train(self, strat_train, outliers_threshold, resampling_method, scaling_method):
        """Wrapper function to process the train set:
            1. Removes outliers if threshold != 0
            2. Resamples if specified.
            3. Drops cluster information columns (not useful for ML).
            4. Fits the scaler
            5. Imputes and splits in X and y.

        Parameters
        ----------
        strat_train : pd.DataFrame
             train set (after stratified split)
        outliers_threshold : int
            Threshold of cluster members for which a molecule will be
            considered an outlier,
        resampling_method : str
            "undersampling", "oversampling" or "none".
        scaling_method : str
            "normalize", "standarize" or "none".

        Returns
        -------
        X_train: pd.DataFrame
        y_train: np.array
        discarded: pd.DataFrame
            outliers + removed molecules if "undersampling".
        fitted_scaler: sklearn.Scaler
            Scaler fitted to the train set. 
        """
        train_clean, outliers = self._remove_outliers(strat_train, n_members_threshold=outliers_threshold)
        train_resampled, waste = self._resample(train_clean, method=resampling_method)
        discarded = outliers.append(waste)
        self._drop_columns(train_resampled, ["cluster_id", "cluster_members", "is_centroid"])
        fitted_scaler = self._fit_scaler(train_resampled, method=scaling_method)
        X_train, y_train = self._clean_to_Xy(train_resampled, fitted_scaler)
        return X_train, y_train, discarded, fitted_scaler

    def _stratified_split(self, df) -> pd.DataFrame:
        """Performs a stratified split (same proportion of actives/inactives
        in the test and train) of the supplied df.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to split (features and labels).

        Returns
        -------
        pd.DataFrame, pd.DataFrame
            trainset, testset
        """
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.test_prop, random_state=42)
        for train_index, test_index in splitter.split(df, df[self.y_col_name]):
            strat_train_set = df.loc[train_index]
            strat_test_set = df.loc[test_index]
        return strat_train_set, strat_test_set

    def _remove_outliers(self, df, n_members_threshold: int = 10, col_name="cluster_members"):
        """Splits the given df (X_train) in regulars and outliers, 
        using the supplied outlier threshold.       

        Parameters
        ----------
        df : pd.DataFrame
            Containing clustering info, features and labels
        n_members_threshold : int, optional
            Threshold of cluster members for which a molecule will be
            considered an outlier, by default 10
        col_name : str, optional
            Column name of the cluster members, by default "cluster_members"

        Returns
        -------
        pd.DataFrame, pd.DataFrame
            regulars df, outliers df
        """
        regulars = df[df[col_name] > n_members_threshold]
        outliers = df[df[col_name] <= n_members_threshold]
        return regulars, outliers

    def _resample(self, df, method: str = "oversampling"):
        """Resamples the given df by the selected method.

        Parameters
        ----------
        df : pd.DataFrame
            Containing the features and labels.
        method : str, optional
            Method to resample: "oversampling", "undersampling" or "none", 
            by default "oversampling"

        Returns
        -------
        pd.DataFrame, pd.DataFrame
            resampled: df after resampling, waste: removed samples if 
            undersampling, empty otherwise.

        Raises
        ------
        BalancingError
            If selected method is not implemented.
        """
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
        """Imputes the given features.

        Parameters
        ----------
        X : pd.DataFrame
            df containing the features.

        Returns
        -------
        pd.DataFrame
            Imputed df.
        """
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        X_imp.index = X.index
        return X_imp
    
    def _fit_scaler(self, train_set, method="standarize"):
        """Fits the scaler to the X_train set.

        Parameters
        ----------
        train_set : pd.DataFrame
            Containing the features and labels.
        method : str, optional
            "normalize", "standarize" or "none", by default "standarize"

        Returns
        -------
        sklearn.scaler
            Fitted scaler to the X_train.

        Raises
        ------
        ScalingError
            If the selected method is not implemented.
        """
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
        """Scales the features of a given X.

        Parameters
        ----------
        X : pd.DataFrame
            df containing only features
        fitted_scaler : sklearn.Scaler
            Scaler fitted to the train dataset.

        Returns
        -------
        pd.DataFrame
            df scaled by the user selected method.
        """
        if fitted_scaler is None:
            return X
        else:
            X_sc = pd.DataFrame(fitted_scaler.transform(X), columns=X.columns)
            X_sc.index = X.index
            return X_sc


    def _clean_to_Xy(self, df, fitted_scaler):
        """Splits a df to X and y (imputing and scaling).

        Parameters
        ----------
        df : pd.DataFrame
            df containing n features and the labels.
        fitted_scaler : sklearn.Scaler
            Scaler fitted to the train dataset.

        Returns
        -------
        pd.Dataframe, np.array
            X, y prepared for ML.
        """
        X = df.drop([self.y_col_name], axis=1)
        X_imp = self._impute(X)
        X_sc = self._scale(X_imp, fitted_scaler)
        
        y = df[self.y_col_name]
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        return X_sc, y_enc
        
    @staticmethod
    def _drop_columns(df, cols_to_remove):
        """Drops columns from the dataframe INPLACE!!!

        Parameters
        ----------
        df : pd.DataFrame
            df to from columns from.
        cols_to_remove : List[str]
            List of columns to drop.
        """
        for column in cols_to_remove:
            if column in list(df.columns):
                df.drop([column], axis=1, inplace=True)

    
