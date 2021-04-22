import os
import pytest

from modtox.ML import preprocess


data_dir = "data"
sdf_active_train = os.path.join(data_dir, "actives.sdf")
sdf_inactive_train = os.path.join(data_dir, "inactives.sdf")
csv_external = os.path.join(data_dir, "glide_features.csv")
csv = None


@pytest.mark.parametrize(
    ("name", "fp", "descriptors", "maccs"),
    [
        ("daylight", True, False, False),
        ("descriptors", False, True, False),
        ("maccs", False, False, True),
    ],
)
def test_feature_extraction(name, fp, descriptors, maccs):
    """
    Tests extraction of all features - external CSV, fingerprints, MACCS keys, descriptors...
    """

    preprocessor = preprocess.ProcessorSDF(
        csv=csv_external, fp=fp, descriptors=descriptors, MACCS=maccs, columns=None, label=None
    )

    X, y = preprocessor.fit_transform(
        sdf_active=sdf_active_train, sdf_inactive=sdf_inactive_train
    )

    X, y, _, y_removed, X_removed, cv = preprocessor.sanitize(X, y, cv=0)
    preprocessor.filter_features(X)

    assert len(y_removed) == len(X_removed) == 33
    assert len(X) == len(y) == 195
