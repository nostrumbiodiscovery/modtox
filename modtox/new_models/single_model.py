import argparse

from numpy.lib.npyio import load
from os import RTLD_NOW
import numpy as np
import pandas as pd
import random
from math import floor

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


INTERNAL_PROPORTION = 0.3
EXTERNAL_PROPORTION = 0.05
ACTIVITY_COLUMN = "Activity"

def generate_single_model(csv, user_model, int_prop=INTERNAL_PROPORTION, ext_prop=EXTERNAL_PROPORTION):
    
    print(f"Pre-processing {csv} (dropping columns, imputation and splitting into sets)... ", end="")
    df = pd.read_csv(csv, index_col=0)

    main_df, external_df, train, int, ext = preprocess(df, int_prop, ext_prop)
    print("Done.")

    model = fit_model(user_model, train)
    y_pred_int = model.predict(int[0])
    y_pred_ext = model.predict(ext[0])

    acc_int, conf_int = score_set(int[1], y_pred_int)
    print(f"Internal test accuracy: {acc_int}")
    print(f"Internal test confusion matrix: {conf_int}")

    acc_ext, conf_ext = score_set(ext[1], y_pred_ext)
    print(f"External test accuracy: {acc_ext}")
    print(f"External test confusion matrix: {conf_ext}")

    return main_df, external_df, acc_int, conf_int, acc_ext, conf_ext


def preprocess(df, int_prop=INTERNAL_PROPORTION, ext_prop=EXTERNAL_PROPORTION):
    df = load_model_input(df)

    main_df, external_df = extract_external_val(df, ext_prop)
    X, y = labelXy(main_df)
    X_ext, y_ext = labelXy(external_df)

    X_imp = imputation(X)
    X_ext_imp = imputation(X_ext)

    X_train, X_test, y_train, y_test = split(X_imp, y, int_prop)
    
    return main_df, external_df, (X_train, y_train), (X_test, y_test), (X_ext_imp, y_ext)

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

def fit_model(user_model, XY_train):
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

    return df, external_df

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

def split(X, y, prop):
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=prop, random_state=42
    )

    return X_train, X_test, y_train, y_test

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
