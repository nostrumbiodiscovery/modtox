import numpy as np
import os
import pickle
import shutil
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import ensemble

import modtox.ML.preprocess as Pre
import modtox.ML.postprocess as Post
import modtox.ML.model2 as model
import modtox.data.pubchem as pchm
import modtox.data.bindingdb as bdb
import modtox.data.dude as dd
import modtox.docking.glide.analyse as gl
import modtox.docking.greasy.greasy_preparation as gre


def create_tmp_dir(tmp):
    if not os.path.exists(tmp): os.mkdir(tmp)


def retrieve_data(size=1500, split=True):
    X, y = datasets.make_blobs(n_samples=size, random_state=170, centers=2)
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return X, y


def compare_models(file1, file2):
    pipe1 = pickle.load(open(file1, 'rb'))
    pipe2 = pickle.load(open(file2, 'rb'))
    comp = []
    for i in pipe1[0].__dict__.keys():
        x1 = pipe1[0].__dict__[i]
        x2 = pipe2[0].__dict__[i]
        assert np.array_equal(x1, x2)


def analyze(inp_files, glide_dir, best, csv, active, inactive):
    gl.analyze(inp_files, glide_dir=glide_dir, best=best, csv=csv, active=active, inactive=inactive, debug=True)


def greasy(active, inactive, systems, folder):
    return gre.GreasyObj(folder=folder, active=active, inactive=inactive, systems=systems)


def retrieve_preprocessor(csv=None, fp=False, descriptors=False, MACCS=True, columns=None):
    return Pre.ProcessorSDF(csv=csv, fp=fp, descriptors=descriptors, MACCS=MACCS, columns=columns, label=None, debug=True)


def retrieve_model(clf, folder, tpot=None, cv=5, generations=None, random_state=42, population_size=None):
    return model.GenericModel(clf=clf, tpot=tpot, cv=cv, generations=generations, random_state=random_state,
                              population_size=population_size, folder=folder)


def retrieve_database_pubchem(pubchem, substrate, nmols, tmp):
    return pchm.PubChem(pubchem=pubchem, train=True, test=False, folder_output=tmp, substrate=substrate,
                        n_molecules_to_read=nmols, debug=True)


def retrieve_database_dude(dude, tmp):
    return dd.DUDE(dude_folder=dude, folder_output=tmp, train=True, test=False, debug=True)


def retrieve_database_bindingdb(binding, tmp):
    return bdb.BindingDB(bindingdb=binding, folder_output=tmp, train=True, test=False, debug=True)


def retrieve_imputer():
    return model.Imputer(imputer_type='cluster_based', n_clusters=1)


def retrieve_processor(train_data=False, clf_stacked=False, folder='.'):
    X_train, X_test, y_train, y_test = retrieve_data()
    clf = ensemble.RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    if clf_stacked:
        clf = 'stack'
        y_pred_clfs = np.array([y_pred, y_pred])
        return Post.PostProcessor(clf, X_test, y_test, y_pred, y_proba, y_pred_test_clfs=y_pred_clfs, x_train=X_train,
                                  y_true_train=y_train, folder=folder)
    if train_data:
        clf = 'single'
        return Post.PostProcessor(clf, X_test, y_test, y_pred, y_proba, x_train=X_train, y_true_train=y_train,
                                  folder=folder)
    else:
        clf = 'single'
        return Post.PostProcessor(clf, X_test, y_test, y_pred, y_proba, folder=folder)


def clean(data):
    if isinstance(data, list):
        for i in data:
            if os.path.exists(i):
                os.remove(i)
    else:
        if os.path.exists(data):
            os.remove(data)


def check_file(path, expected_lines):
    """
    Checks if expected lines are present in the file. Returns a list of lines which were not found.
    """
    errors = []

    with open(path, "r") as f:
        content = f.read()

    for line in expected_lines:
        if line not in content:
            errors.append(line)

    return errors


def check_remove_folder(*output_folders):
    """
    Removes the whole folder tree.

    Parameters
    ----------
    output_folders : Union[str, List[str]]
        Path(s) to folder to be removed
    """
    for folder in output_folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
