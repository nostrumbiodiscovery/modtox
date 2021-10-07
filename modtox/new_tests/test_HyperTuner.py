from pytest import approx
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import VotingClassifier
from modtox.modtox.new_models_classes.ML.tuning import GridHalvingSearch, RandomHalvingSearch, RandomSearch

X, y = load_breast_cancer(return_X_y=True)

def test_tuners():
    tuner = RandomSearch(X, y)
    votclf = tuner.search(random_state=42)
    assert isinstance(votclf, VotingClassifier)
    assert tuner.score == approx(0.93, abs=0.1)
    assert isinstance(tuner.best_params, dict)

def test_random_halving():
    """Pretty quick"""
    tuner = RandomHalvingSearch(X, y)
    votclf = tuner.search(random_state=42)
    assert isinstance(votclf, VotingClassifier)
    assert tuner.score == approx(0.96, abs=0.1)
    assert isinstance(tuner.best_params, dict)

def test_grid_halving():
    """Really slow"""
    tuner = GridHalvingSearch(X, y)
    votclf = tuner.search(random_state=42)
    assert isinstance(votclf, VotingClassifier)
    assert tuner.score == approx(0.96, abs=0.1)
    assert isinstance(tuner.best_params, dict)