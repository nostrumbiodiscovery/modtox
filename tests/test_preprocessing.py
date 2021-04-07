import os
import pytest

from modtox.ML import preprocess

data_dir = "data"
sdf_active_train = os.path.join(data_dir, "actives.sdf")
sdf_inactive_train = os.path.join(data_dir, "inactives.sdf")
csv = os.path.join(data_dir, "glide_features.csv")


@pytest.mark.parametrize(
    ("fp", "descriptors", "maccs"),
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_feature_extraction(fp, descriptors, maccs):
    """
    Tests extraction of all features - external CSV, fingerprints, MACCS keys, descriptors...
    """
    preprocessor = preprocess.ProcessorSDF(
        csv=csv, fp=fp, descriptors=descriptors, MACCS=maccs, columns=None, label=None
    )
    X, y = preprocessor.fit_transform(
        sdf_active=sdf_active_train, sdf_inactive=sdf_inactive_train
    )
    preprocessor.sanitize(X, y, cv=0)
    preprocessor.filter_features(X)

    assert X
