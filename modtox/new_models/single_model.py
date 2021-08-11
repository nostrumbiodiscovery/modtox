import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.impute import SimpleImputer


def run(csv, user_model):
    """
    Load X and y, run simple imputation, then fit and predict single model defined by the user.

    Parameters
    ----------
    csv : str
        Path to CSV file with the final model input.
    user_model : str
        String representing a single model, one of: knn, lr, svc, tree or nb.

    Returns
    -------
        Fitted model.
    """
    from utils.utils import get_data

    X, y = get_data(csv)

    X_imp = imputation(X)
    X_train, X_test, y_train, y_test = split(X_imp, y)

    print("Fitting the model...")
    model = retrieve_model(user_model)
    model.fit(X_train, y_train)

    print("Running prediction on test set...")
    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Confusion matrix: tp", tp, "fp", fp, "fn", fn, "tn", tn)
    print("Final score:", (tp + tn) / (tp + tn + fp + fn))

    return model


def imputation(X):

    imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    X_imp = imputer.fit_transform(X)
    print("Imputation done.")
    return X_imp


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

    print(f"Retrieved model {user_model}.")
    return models[user_model.lower()]


def split(X, y):
    """
    Splits data into train (70%) and test (30%) sets.

    Parameters
    ----------
    X : np.array
        Array containing features.
    y : np.array
        Array containing labels.

    Returns
    -------
        Arrays for X_train, X_test, y_train and y_test.
    """
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
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
    run(args.csv, args.model)
