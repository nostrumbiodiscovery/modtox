import pytest
import numpy as np
import os
import pickle
import glob
import filecmp
import tests.config as tc

TMP = os.path.join(os.path.dirname(__file__), "tmp")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
ACTIVE = os.path.join(DATA_PATH, "actives.sdf")
DOCKING = os.path.join(DATA_PATH, "docking")
GLIDE_FILES = os.path.join(DOCKING, "*dock_lib.maegz")
INACTIVE = os.path.join(DATA_PATH, "inactives.sdf")
GLIDE_FEATURES = os.path.join(DATA_PATH, "glide_features.csv")
PUBCHEM = os.path.join(DATA_PATH, "AID_1851_datatable_all.csv")
SYSTEMS = glob.glob(os.path.join(DOCKING, "*.zip*"))
DUDE = os.path.join(DATA_PATH, "cp2c9")
BINDING = os.path.join(DATA_PATH, "cyp2c9_bindingdb.sdf")
SUBSTRATE = "p450-cyp2c9"
NMOLS = 10
BEST = False
CSV = False

tc.create_tmp_dir(TMP)


# DOCKING TESTS

@pytest.mark.parametrize("glide_files, best, csv, active, inactive", [
    (GLIDE_FILES, BEST, CSV, ACTIVE, INACTIVE),
])
def test_docking(glide_files, best, csv, active, inactive):
    inp_files = glob.glob(os.path.join(DATA_PATH, glide_files))
    tc.analyze(inp_files=inp_files, glide_dir=DOCKING, best=best, csv=csv, active=active, inactive=inactive)


@pytest.mark.parametrize("active, inactive, systems", [
    (ACTIVE, INACTIVE, SYSTEMS),
])
def test_greasy(active, inactive, systems):
    gre = tc.greasy(folder=TMP, active=active, inactive=inactive, systems=systems)
    gre.preparation()
    outputs = glob.glob(os.path.join(TMP, 'input*.in'))
    refs = glob.glob(os.path.join(DATA_PATH, 'input*.in'))
    outputs.sort()
    refs.sort()
    assert False not in [filecmp.cmp(output, ou) for output, ou in zip(outputs, refs)]


# DATABASE TESTS

@pytest.mark.parametrize("dude, tmp", [
    (DUDE, TMP),
])
def test_preparation_dude(dude, tmp):
    dude = tc.retrieve_database_dude(dude=dude, tmp=tmp)
    sdf_active, sdf_inactive = dude.process_dude()
    assert filecmp.cmp(os.path.join(TMP, 'used_mols.txt'), os.path.join(DATA_PATH, 'used_mols_dude.txt'))


@pytest.mark.parametrize("substrate, pubchem, nmols, tmp", [
    (SUBSTRATE, PUBCHEM, NMOLS, TMP),
])
def test_preparation_pubchem(substrate, pubchem, nmols, tmp):
    pubchem = tc.retrieve_database_pubchem(substrate=substrate, pubchem=pubchem, nmols=nmols, tmp=tmp)
    sdf_active, sdf_inactive = pubchem.process_pubchem(readfromfile=False)
    assert filecmp.cmp(os.path.join(TMP, 'used_mols.txt'), os.path.join(DATA_PATH, 'used_mols_pubchem.txt'))


@pytest.mark.parametrize("binding, tmp", [
    (BINDING, TMP),
])
def test_preparation_bindingdb(binding, tmp):
    binding = tc.retrieve_database_bindingdb(binding=binding, tmp=tmp)
    sdf_active, sdf_inactive = binding.process_bind()
    assert filecmp.cmp(os.path.join(TMP, 'used_mols.txt'), os.path.join(DATA_PATH, 'used_mols_bindingdb.txt'))


# PREPROCESS TESTS
@pytest.mark.parametrize("sdf_active, sdf_inactive, glide_features", [
    (ACTIVE, INACTIVE, GLIDE_FEATURES),
])
def test_preprocess_fit_transform_all(sdf_active, sdf_inactive, glide_features):
    pre = tc.retrieve_preprocessor(csv=glide_features, fp=True, descriptors=True, MACCS=True, columns=None)
    X_trans, _ = pre.fit_transform(sdf_active=sdf_active, sdf_inactive=sdf_inactive, folder=TMP)
    X = np.array(X_trans)
    np.save(os.path.join(TMP, 'preprocess_all'), X)
    assert filecmp.cmp(os.path.join(TMP, 'preprocess_all.npy'), os.path.join(DATA_PATH, 'preprocess_all.npy'))


@pytest.mark.parametrize("sdf_active, sdf_inactive, glide_features", [
    (ACTIVE, INACTIVE, GLIDE_FEATURES),
])
def test_preprocess_fit_transform_glide(sdf_active, sdf_inactive, glide_features):
    pre = tc.retrieve_preprocessor(csv=glide_features, fp=False, descriptors=False, MACCS=False, columns=None)
    X_trans, _ = pre.fit_transform(sdf_active=sdf_active, sdf_inactive=sdf_inactive, folder=TMP)
    X = np.array(X_trans)
    np.savetxt(os.path.join(TMP, 'preprocess_glide.out'), X, delimiter=',')
    assert filecmp.cmp(os.path.join(TMP, 'preprocess_glide.out'), os.path.join(DATA_PATH, 'preprocess_glide.out'))


@pytest.mark.parametrize("sdf_active, sdf_inactive, glide_features", [
    (ACTIVE, INACTIVE, GLIDE_FEATURES),
])
def test_preprocess_sanitize(sdf_active, sdf_inactive, glide_features):
    pre = tc.retrieve_preprocessor(csv=glide_features)
    X, y = pre.fit_transform(sdf_active=sdf_active, sdf_inactive=sdf_inactive, folder=TMP)
    X, y, _, _, _, _ = pre.sanitize(X, y, cv=5, folder=TMP)
    X = np.array(X)
    np.savetxt(os.path.join(TMP, 'preprocess_sanitize.out'), X, delimiter=',')
    assert filecmp.cmp(os.path.join(TMP, 'preprocess_sanitize.out'), os.path.join(DATA_PATH, 'preprocess_sanitize.out'))


@pytest.mark.parametrize("sdf_active, sdf_inactive, glide_features", [
    (ACTIVE, INACTIVE, GLIDE_FEATURES),
])
def test_preprocess_filter_features(sdf_active, sdf_inactive, glide_features):
    pre = tc.retrieve_preprocessor(csv=glide_features, columns=['rdkit_fingerprintMACS_5', 'rdkit_fingerprintMACS_50'])
    X, y = pre.fit_transform(sdf_active=sdf_active, sdf_inactive=sdf_inactive, folder=TMP)
    X, y, _, _, _, _ = pre.sanitize(X, y, cv=5, folder=TMP)
    import pdb;
    pdb.set_trace()
    X, _ = pre.filter_features(X)
    X = np.array(X)
    np.savetxt(os.path.join(TMP, 'preprocess_filter.out'), X, delimiter=',')
    assert filecmp.cmp(os.path.join(TMP, 'preprocess_filter.out'), os.path.join(DATA_PATH, 'preprocess_filter.out'))


# MODEL TESTS

def test_model_stack():
    X_train, X_test, y_train, y_test = tc.retrieve_data()
    model = tc.retrieve_model(clf="stack", cv=5, folder=TMP)
    model.fit(X_train, y_train)
    data1 = [];
    data2 = []
    f1 = open(os.path.join(TMP, 'opt_model.pkl'), 'rb')
    f2 = open(os.path.join(DATA_PATH, 'model_fit_stack.pkl'), 'rb')
    while True:
        try:
            data1.append(pickle.load(f1))
            data2.append(pickle.load(f2))
        except EOFError:
            break
    predictions1 = [cl.predict(X_test) for cl in data1[0:-1]]
    predictions2 = [cl.predict(X_test) for cl in data2[0:-1]]
    assert np.array_equal(predictions1, predictions2)
    assert False not in model.predict(X_test, y_test)


def test_model_single():
    X_train, X_test, y_train, _ = tc.retrieve_data()
    model = tc.retrieve_model(clf="single", cv=5, folder=TMP)
    model.fit(X_train, y_train)
    data1 = []
    data2 = []
    f1 = open(os.path.join(TMP, 'opt_model.pkl'), 'rb')
    f2 = open(os.path.join(DATA_PATH, 'model_fit_single.pkl'), 'rb')
    while True:
        try:
            data1.append(pickle.load(f1))
            data2.append(pickle.load(f2))
        except EOFError:
            break
    preds = [cl1.predict(X_test) == cl2.predict(X_test) for cl1, cl2 in zip(data1, data2)]
    assert False not in np.stack(preds)


def test_model_fit_stack_tpot():
    X_train, _, y_train, _ = tc.retrieve_data()
    model = tc.retrieve_model(clf="stack", tpot=True, cv=5, generations=1, population_size=5, random_state=42,
                              folder=TMP)
    model.fit(X_train, y_train)
    tc.compare_models(os.path.join(TMP, 'opt_model.pkl'), os.path.join(DATA_PATH, 'model_fit_stack_tpot.pkl'))


def test_model_fit_single_tpot():
    X_train, _, y_train, _ = tc.retrieve_data()
    model = tc.retrieve_model(clf="single", tpot=True, cv=5, generations=1, population_size=5, random_state=42,
                              folder=TMP)
    model.fit(X_train, y_train)
    tc.compare_models(os.path.join(TMP, 'opt_model.pkl'), os.path.join(DATA_PATH, 'model_fit_single_tpot.pkl'))


def test_imputer():
    X_train, _, _, _ = tc.retrieve_data()
    imputer = tc.retrieve_imputer()
    imputer.fit_transform(X_train)
    np.savetxt(os.path.join(TMP, 'test_imputer.out'), imputer.imputer.xmeans, delimiter=',')
    assert filecmp.cmp(os.path.join(TMP, 'test_imputer.out'), os.path.join(DATA_PATH, 'imputer.out'))


# POST-PROCESSING TESTS

def test_ROC(output="ROC.png"):
    tc.clean(os.path.join(TMP, output))
    pp = tc.retrieve_processor(folder=TMP)
    ROC = pp.ROC(output_ROC=output)
    assert os.path.exists(os.path.join(TMP, output))


def test_PR(output="PR.png"):
    tc.clean(os.path.join(TMP, output))
    pp = tc.retrieve_processor(folder=TMP)
    PR = pp.PR(output_PR=output)
    assert os.path.exists(os.path.join(TMP, output))


def test_shap(output='shap.png'):
    tc.clean(os.path.join(TMP, output))
    pp = tc.retrieve_processor(train_data=True, folder=TMP)
    SH = pp.shap_values(output_shap=output, debug=True)
    assert os.path.exists(os.path.join(TMP, output))


def test_distributions(output='distributions'):
    tc.clean(glob.glob(output + "*"))
    pp = tc.retrieve_processor(train_data=True, folder=TMP)
    DD = pp.distributions(output_distributions=output)
    assert any(glob.glob(os.path.join(TMP, 'distributions_*')))


def test_feature_importance(output='feature_importance.txt'):
    tc.clean(os.path.join(TMP, output))
    pp = tc.retrieve_processor(train_data=True, folder=TMP)
    FI = pp.feature_importance(output_features=output)
    assert os.path.exists(os.path.join(TMP, output))


def test_domain_analysis(output_densities="thresholds_vs_density.png", output_thresholds="threshold_analysis.txt",
                         output_distplots="distplot"):
    output_densities = os.path.join(TMP, output_densities)
    output_thresholds = os.path.join(TMP, output_thresholds)
    output_distplots = os.path.join(TMP, output_distplots)
    output = [output_densities, output_thresholds] + glob.glob(output_distplots + "*")
    tc.clean(output)
    pp = tc.retrieve_processor(train_data=True)
    DA = pp.domain_analysis(output_densities=output_densities, output_thresholds=output_thresholds,
                            output_distplots=output_distplots, debug=True)
    assert any(output)


def test_confusion_matrix(output="confusion_matrix.png"):
    tc.clean(os.path.join(TMP, output))
    pp = tc.retrieve_processor(folder=TMP)
    DA = pp.conf_matrix(output_conf=output)
    assert any(os.path.join(TMP, output))


def test_uncertainties():
    pp = tc.retrieve_processor(train_data=True, clf_stacked=True, folder=TMP)
    UN = pp.calculate_uncertanties()


def test_umap(output_umap="umap_proj"):
    tc.clean(glob.glob(os.path.join(TMP, output_umap) + "*"))
    pp = tc.retrieve_processor(folder=TMP, train_data=True)
    pp.UMAP_plot(output_umap=output_umap, single=True, wrong=True, wrongall=True, traintest=True, wrongsingle=True)
    assert any(os.path.join(TMP, output_umap))


def test_pca(output_pca="pca.png"):
    tc.clean(os.path.join(TMP, output_pca))
    pp = tc.retrieve_processor(folder=TMP)
    pp.PCA_plot(output_pca=output_pca)
    assert any(os.path.join(TMP, output_pca))


def test_tsne(output_tsne="tsne.png"):
    tc.clean(os.path.join(TMP, output_tsne))
    pp = tc.retrieve_processor(folder=TMP)
    pp.tsne_plot(output_tsne=output_tsne)
    assert any(os.path.join(TMP, output_tsne))
