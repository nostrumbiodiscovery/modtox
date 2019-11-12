import numpy as np
import os
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import ensemble
import modtox.ML.preprocess as Pre
import modtox.ML.postprocess as Post
import modtox.ML.model2 as model
import modtox.data.pubchem as pchm
import modtox.data.dude as dd
import modtox.docking.glide.analyse as gl
import modtox.docking.greasy.greasy_preparation as gre

def create_tmp_dir(tmp):
    if not os.path.exists(tmp): os.mkdir(tmp)

def retrieve_data(size=1500, split=True):
    X, y = datasets.make_blobs(n_samples=size, random_state=170, centers=2)
    if split==True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test   
    else:
        return X,y

def analyze(inp_files, glide_dir, best, csv, active, inactive):
  
    gl.analyze(inp_files, glide_dir=glide_dir, best=best, csv=csv, active=active, inactive=inactive, debug=True)

def greasy(active, inactive, systems, folder):

    return gre.GreasyObj(folder=folder, active=active, inactive=inactive, systems=systems)

def retrieve_preprocessor(csv=None, fp=False, descriptors=False, MACCS=True, columns=None):
    return Pre.ProcessorSDF(csv=csv, fp=fp, descriptors=descriptors, MACCS=MACCS, columns=columns, debug=True)

def retrieve_model(clf, folder, tpot=None):
    return model.GenericModel(clf=clf, tpot=tpot, folder=folder)

def retrieve_database_pubchem(pubchem, substrate, nmols, tmp):
    return pchm.PubChem(pubchem=pubchem, train=True, test=False, folder_output=tmp, substrate=substrate, n_molecules_to_read=nmols, debug=True)

def retrieve_database_dude(dude, tmp):
    return dd.DUDE(dude_folder=dude, folder_output=tmp, train=True, test=False, debug=True)

def retrieve_imputer():
    return model.Imputer(imputer_type='cluster_based', n_clusters=1)

def retrieve_processor(train_data=False, clf_stacked=False, folder='.'):

    X_train, X_test, y_train, y_test = retrieve_data()
    clf = ensemble.RandomForestClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    if clf_stacked: 
        clf = 'stack'
        y_pred_clfs = np.array([y_pred, y_pred])
        return Post.PostProcessor(clf, X_test, y_test, y_pred, y_proba, y_pred_test_clfs=y_pred_clfs, x_train=X_train, y_true_train=y_train, folder=folder)
    if train_data:
        clf = 'single'
        return Post.PostProcessor(clf, X_test, y_test, y_pred, y_proba, x_train=X_train, y_true_train=y_train, folder=folder)
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

