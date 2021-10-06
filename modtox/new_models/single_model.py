"""
--------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------
This script builds and tests the model from the balanced Glide features with or without
other descriptors. 
    1. Preprocess the feeded dataframe (drop irrelevant columns, imputation, format...)
    2. Fits the data to the model specified.
    3. Scores the fitted model. 

Can be used as individual script or implemented in a pipeline. 
--------------------------------------------------------------------------------------------
USAGE (as script)*
python single_model.py --csv glide_mordred.csv --model knn
    * Both arguments are required. 
--------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------
"""

import argparse

import numpy as np
import pandas as pd
import random
from math import floor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from modtox.modtox.new_models._parameters import *

"""
--------------------------------------------------------------------------------------------
MAIN FUNCTION
--------------------------------------------------------------------------------------------
"""
def generate_single_model(csv, user_model, test_prop=INTERNAL_PROPORTION, ext_prop=EXTERNAL_PROPORTION):
    """
    Fits the supplied data into the model specified by the user. 
    
    Parameters 
    ----------
    csv : str 
        Path to CSV file with any combination of data from three dataframes: 
        Glide features, mordred descriptors and topological fingerprints.
    
    user_model : str
        User selected model between knn, lr, svc, tree or nb.

    test_prop : float, default = 0.3 
        Proportion of data used to test the model. Default is set at 
        '_parameters.py' file.

    ext_prop : float, default = 0.05
        Proportion of external set extracted to validate the model. 
        Default is set at '_parameters.py' file.

    Returns
    -------

    """
    # Read CSV and format pandas dataframe
    print(f"Pre-processing {csv} (dropping columns, imputation and splitting into sets)... ", end="")
    df = pd.read_csv(csv, index_col=0)
    df = load_model_input(df)

    # Extract 5% external validation, label and imputation
    main_df, external_df, X, y, X_ext, y_ext = extract_external_val(df, ext_prop)
    
    # Retrieve model from user
    model = retrieve_model(user_model)

    # Method 1: from the 95% remaining 30% test, 70% train.
        # Fits model to train set
        # Predicts with test set
        # Predicts with external data set
    acc_test, conf_test, acc_ext, conf_ext = train_test_external(model, X, y, X_ext, y_ext)

    # Method 2: 5% external validation, 95% cross-validation (k=5). 
    # Only extracts scoring. None of the built models has seen the
    # external data.
    cv_score = cross_validation(model, X, y)
    
    print(f"Internal test accuracy: {acc_test}")
    print(f"Internal test confusion matrix: {conf_test}")
    print(f"External test accuracy: {acc_ext}")
    print(f"External test confusion matrix: {conf_ext}")
    print(f"Cross validation score: {cv_score}")
    
    return main_df, external_df, acc_test, conf_test, acc_ext, conf_ext, cv_score

"""
--------------------------------------------------------------------------------------------
HELPER FUNCTIONS
--------------------------------------------------------------------------------------------
"""    

def retrieve_model(user_model):
    """
    Retrieves classifier and its parameters based on the selection supplied by the user.

    Parameters
    ----------
    user_model : str
        String representing a single model, one of: knn, lr, svc, tree or nb.

    Returns
    -------
        Classifier object.
    """
    models = {
        "knn": KNeighborsClassifier(n_neighbors=2, p=1, weights="distance"),
        "svc": SVC(C=0.1, class_weight="balanced", kernel="rbf", tol=0.01),
        "lr": LogisticRegression(
            C=0.01, class_weight="balanced", dual=False, penalty="l2", max_iter=5000
        ),
        "tree": DecisionTreeClassifier(
            class_weight="balanced",
            criterion="entropy",
            max_depth=3,
            min_samples_leaf=1,
            min_samples_split=12,
            splitter="random",
        ),
        "nb": BernoulliNB(alpha=0.001, fit_prior=False),
    }

    # print(f"Retrieved model {user_model}.")
    return models[user_model.lower()]

def fit_model(user_model, X_train, y_train):
    model = retrieve_model(user_model)
    model.fit(XY_train[0], XY_train[1])
    return model

def score_set(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    conf_matrix = f"tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}"
    #print("Confusion matrix:", conf_matrix)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    #print("Final score:", accuracy)
    return accuracy, conf_matrix

def load_model_input(df):
  
    COLUMNS_TO_EXCLUDE = ["Title"]
    # Drop irrelevant columns
    numbered_columns = [column for column in df.columns if "#" in column]

    for column in numbered_columns + COLUMNS_TO_EXCLUDE:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)
    
    # Drop NaN columns
    df.dropna(axis="columns", how="all", inplace=True)

    # Replace NaN with 0
    df.fillna(0, inplace=True)
    return df

def extract_external_val(df, prop):
    """
    Extracts from <rdkit.obj> list, not dataframe. 
    1.  Calculates size of external set as a % of balanced set.
    2.  Selects from actives/inactives (different randint to 
        avoid correlation, if exists).
    3.  Removes random item and appends to new dataframe 
    -----------------------------------------------------------
    INP:    actives, inactives lists.
    OUT:    external_set <rdkit.obj> list. Modifies act/inact lists
            in place.
    """
    
    rows = df.shape[0]
    # Define size of external validation set 
    # (even to avoid re-balancing)
    size = floor(rows * prop/2)
    if not size % 2 == 0:
        size = size - 1
    
    external_df = pd.DataFrame(columns=df.columns)

    for _ in range(0, size):
        act_col = df[ACTIVITY_COLUMN]
        all_idxs = list(df.index)
        actives_idxs = list(df.iloc[act_col.to_list()].index)
        inactives_idxs = [ idx for idx in all_idxs if idx not in actives_idxs ]

        rdm_idxs = (random.choice(actives_idxs), random.choice(inactives_idxs))
        
        for idx in rdm_idxs:
            external_df = external_df.append(df.loc[idx])
            df.drop(index=idx, inplace=True)

    # Label and imputation
    X, y = labelXy(df)
    X_ext, y_ext = labelXy(external_df)
    X = imputation(X)
    X_ext = imputation(X_ext)
    main_df = df
    return main_df, external_df, X, y, X_ext, y_ext

def labelXy(df):
    activity_column = "Activity"    
    # Assign labels column to y and drop it from the main dataframe
    y = df[activity_column]
    df.drop(activity_column, axis=1, inplace=True)

    # Convert labels from True/False to 0/1
    le = LabelEncoder()
    y_trans = le.fit_transform(y)

    X = df
    #X.to_csv("X.csv")
    
    # print(f"Finished processing the input data. X {X.shape}, y {y_trans.shape}")
    #np.save("X.npy", X)
    #np.save("y.npy", y_trans)

    return X, y_trans

def imputation(X):

    imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    X_imp = imputer.fit_transform(X)
    # print("Imputation done.")
    return X_imp

def split(X, y, prop=INTERNAL_PROPORTION):
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=prop, random_state=42
    )

    return X_train, X_test, y_train, y_test

def cross_validation(model, X, y):
    scores = cross_val_score(model, X, y)
    mean = "%.2f" % float(scores.mean())
    std = "%.2f" % float(scores.std())
    cv_score = f"{mean} +/- {std}"
    return cv_score

def train_test_external(model, X, y, X_ext, y_ext):
    X_train, X_test, y_train, y_test = split(X, y)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc_test, conf_test = score_set(y_test, y_pred)
    
    y_ext_pred = model.predict(X_ext)
    acc_ext, conf_ext = score_set(y_ext, y_ext_pred)
    return acc_test, conf_test, acc_ext, conf_ext

"""
--------------------------------------------------------------------------------------------
ARGPARSE
--------------------------------------------------------------------------------------------
"""
def parse_args(parser):
    """
    Parses command line arguments.

    Parameters
    ----------
    parser : ArgumentParser
        Object initialized from argparse.
    """
    parser.add_argument("--csv", help="CSV with the final model input.")
    parser.add_argument(
        "--model", help="Choose one of the available models: knn, lr, svc, tree or nb."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build single classification models by supplying model_input.csv and model type (knn, lr, svc, "
        "tree or nb). "
    )
    parse_args(parser)
    args = parser.parse_args()
    generate_single_model(args.csv, args.model)
