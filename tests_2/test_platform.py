import pytest
import os
import glob
import test_config as tc


DATA_PATH=os.path.join(os.path.dirname(__file__), "data")
ACTIVE=os.path.join(DATA_PATH, "actives.sdf")
INACTIVE=os.path.join(DATA_PATH, "inactives.sdf")
GLIDE_FEATURES=os.path.join(DATA_PATH, "glide_features.csv")


########## PREPROCESS TESTS  #####################
@pytest.mark.parametrize("sdf_active, sdf_inactive", [
                         (ACTIVE, INACTIVE),
                         ])
def test_preprocess_fit_transform(sdf_active, sdf_inactive):
    pre = tc.retrieve_preprocessor()
    pre.fit_transform(sdf_active=sdf_active, sdf_inactive=sdf_inactive)

@pytest.mark.parametrize("sdf_active, sdf_inactive, glide_features", [
                         (ACTIVE, INACTIVE, GLIDE_FEATURES),
                         ]) 
def test_preprocess_sanitize(sdf_active, sdf_inactive, glide_features):
 
    pre = tc.retrieve_preprocessor(csv=glide_features)
    X,y = pre.fit_transform(sdf_active=sdf_active, sdf_inactive=sdf_inactive)
    pre.sanitize(X, y)

@pytest.mark.parametrize("sdf_active, sdf_inactive, glide_features", [
                         (ACTIVE, INACTIVE, GLIDE_FEATURES),
                         ]) 
def test_preprocess_filter_features(sdf_active, sdf_inactive, glide_features):

    pre = tc.retrieve_preprocessor(csv=glide_features, columns=['rdkit_fingerprintMACS_5', 'rdkit_fingerprintMACS_50'])
    X, y = pre.fit_transform(sdf_active=sdf_active, sdf_inactive=sdf_inactive)
    pre.sanitize(X, y)
    pre.filter_features(X)

########## MODEL TESTS #####################

def test_model_fit_stack():
    model = tc.retrieve_model(clf="stack") 
    X_train, _, y_train , _ = tc.retrieve_data()
    model.fit(X_train, y_train)

def test_model_fit_single():

    X_train, _, y_train , _ = tc.retrieve_data()
    model = tc.retrieve_model(clf="single")
    model.fit(X_train, y_train)

def test_model_fit_stack_tpot():

    X_train, _, y_train , _ = tc.retrieve_data()
    model = tc.retrieve_model(clf="stack", tpot=True)
    model.fit(X_train, y_train)

def test_model_fit_single_tpot():

    X_train, _, y_train , _ = tc.retrieve_data()
    model = tc.retrieve_model(clf="single", tpot=True)
    model.fit(X_train, y_train)

def test_imputer():
    X_train, _, _ , _ = tc.retrieve_data()
    imputer = tc.retrieve_imputer()
    imputer.fit_transform(X_train)

#############  POST-PROCESSING TESTS #################

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
    SH = pp.shap_values(output_shap=output, debug=True)
    assert os.path.exists(output)

def test_distributions(output='distributions'):
    tc.clean(glob.glob(output+"*"))
    pp = tc.retrieve_processor(train_data=True)
    DD = pp.distributions(output_distributions=output)
    assert any(glob.glob('distributions_*'))

def test_feature_importance(output='feature_importance.png'):
    tc.clean(output)
    pp = tc.retrieve_processor(train_data=True)
    FI = pp.feature_importance(output_features=output)
    assert os.path.exists(output)

def test_domain_analysis(output_densities="thresholds_vs_density.png", output_thresholds="threshold_analysis.txt", output_distplots="displot"):
    output = [output_densities, output_thresholds] + glob.glob(output_distplots+"*")
    tc.clean(output)
    pp = tc.retrieve_processor(train_data=True)
    DA = pp.domain_analysis(output_densities=output_densities, output_thresholds=output_thresholds, output_distplots=output_distplots, debug=True)
    assert any(output)

