from modtox.modtox.ML.selector import _RFECV, _PCA
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
import pandas as pd
import os

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAINSET = os.path.join(DATA, "tests_outputs", "train_set.csv")
df = pd.read_csv(TRAINSET, index_col=0)
X = df.drop("Activity", axis=1)
y = df["Activity"]


def test_rfecv_fit_selector():
    """Tests selector is fitted."""
    selector = _RFECV(step=20)
    fitted_selector = selector.fit_selector(X, y)
    assert isinstance(fitted_selector, RFECV)
    assert hasattr(fitted_selector, "estimator_")


def test_rfecv_plot_data():
    """Tests data is ok for plotting, starting at 1 feature
    (min_features_to_select attribute)"""
    selector = _RFECV(step=20)
    fitted_selector = selector.fit_selector(X, y)
    x_data, y_data = selector.plotting_data()
    assert len(x_data) == len(y_data)
    assert x_data[0] == 1


def test_pca_fit_selector():
    """Tests selector is fitted."""
    selector = _PCA(variance=0.95)
    fitted_selector = selector.fit_selector(X, y)
    assert isinstance(fitted_selector, PCA)
    assert hasattr(fitted_selector, "components_")


def test_pca_plot_data():
    """Tests data is ok for plotting."""
    selector = _PCA(variance=0.95)
    fitted_selector = selector.fit_selector(X, y)
    x_data, y_data = selector.plotting_data()
    assert len(x_data) == len(y_data)
