from modtox.modtox.ML.model import Model, ModelSummary
from modtox.modtox.ML.selector import RFECrossVal
from modtox.modtox.ML.tuning import RandomHalvingSearch

import numpy as np
import os
import pandas as pd

from unittest.mock import patch

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REDUCED_DATASET = os.path.join(DATA, "reduced_dataset")
FULL_DATASET = os.path.join(DATA, "full_dataset")

ACTIVES = os.path.join(REDUCED_DATASET, "actives.sdf")
INACTIVES = os.path.join(REDUCED_DATASET, "inactives.sdf")
GLIDE_CSV = os.path.join(REDUCED_DATASET, "glide_features.csv")

OUTPUT_FOLDER = os.path.join(DATA, "tests_outputs")

train_set_csv = os.path.join(FULL_DATASET, "train_set.csv")
test_set_csv = os.path.join(FULL_DATASET, "test_set.csv")


def to_Xy(csv):
    df = pd.read_csv(csv, index_col=0)
    X = df.drop("Activity", axis=1)
    y = df["Activity"].to_numpy()
    return X, y


X_train, y_train = to_Xy(train_set_csv)
X_test, y_test = to_Xy(test_set_csv)

m = Model(X_train, y_train, X_test, y_test)

# INTEGRATION TEST
def test_build_model():
    summ = m.build_model()
    assert isinstance(summ.selector, RFECrossVal)
    assert isinstance(summ.tuner, RandomHalvingSearch)
    assert isinstance(summ.y_pred, np.ndarray)


@patch("modtox.modtox.ML.selector.RFECrossVal.get_scores", autospec=True)
def test_plot_feature_selection(mock):
    x = sorted(list(range(1, 4001, 10)), reverse=True)
    y = np.random.uniform(low=0.75, high=0.95, size=(400,))
    mock.return_value = x, y
    summ = ModelSummary(RFECrossVal(1000), "tuner", "estimator", "y_pred")
    summ.plot_feature_selection()


def test_get_estimators_accuracy():
    summ = m.build_model(step=1000)
    df = summ.get_estimators_accuracy(m.X_test, m.y_test)
    # df.to_csv('classifier_scores.csv')
    assert df.shape == (6, 6)


"""{'KNeighborsClassifier': 0.8057553956834532, 'SVC': 0.9280575539568345, 'LogisticRegression': 0.9136690647482014, 'DecisionTreeClassifier': 0.9712230215827338, 'BernoulliNB': 0.8633093525179856, 'VotingClassifier': 0.9424460431654677}
"""
