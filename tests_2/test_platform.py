import pytest
import os
import glob
import test_config as tc

def test_ROC(output="ROC.png"):
    tc.clean(output)
    pp = tc.retrieve_processor()
    ROC = pp.ROC(output_ROC=output)
    assert os.path.exists(output)

def test_PR(output="PR.png"):
    tc.clean(output)
    pp = tc.retrieve_processor()
    PR = pp.PR(output_PR=output)
    assert os.path.exists(output)

def test_shap(output='shap.png'):
    tc.clean(output)
    pp = tc.retrieve_processor(train_data=True)
    PR = pp.shap_values(output_shap=output, debug=True)
    assert os.path.exists(output)

def test_distributions(output='distributions'):
    tc.clean(glob.glob(output+"*"))
    pp = tc.retrieve_processor(train_data=True)
    PR = pp.distributions(output_distributions=output)
    assert any(glob.glob('distributions_*'))

def test_feature_importance(output='feature_importance.png'):
    tc.clean(output)
    pp = tc.retrieve_processor(train_data=True)
    PR = pp.feature_importance(output_features=output)
    assert os.path.exists(output)
