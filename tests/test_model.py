import glob
import numpy as np
import os
import pytest
from sklearn.model_selection import train_test_split

from modtox.ML import model2

features_path = os.path.join("data", "features")


@pytest.mark.parametrize(
    ("classifier", "tpot"),
    [("stack", True), ("stack", False), ("single", True), ("single", False)],
)
def test_model_building(classifier, tpot):

    X = np.load(os.path.join(features_path, "X.npy"))
    y = np.load(os.path.join(features_path, "y.npy"))
    X_removed = np.load(os.path.join(features_path, "X_removed.npy"))
    y_removed = np.load(os.path.join(features_path, "y_removed.npy"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    Model = model2.GenericModel(
        X=X_train,
        Y=y_train,
        clf=classifier,
        tpot=tpot,
        X_removed=X_removed,
        y_removed=y_removed,
        generations=1,
    )
    Model.fit(X_train, y_train)

    y_pred = Model.predict(X_test, y_test, X_removed=X_removed, y_removed=y_removed)
    assert y_pred

    output_files = glob.glob("*pkl")
    for file in output_files:
        os.remove(file)
