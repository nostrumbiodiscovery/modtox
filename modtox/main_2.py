import glob
import os
import argparse
import numpy as np
import modtox.ML.preprocess as Pre
import modtox.ML.postprocess as Post
import modtox.ML.model2 as model
import modtox.Helpers.helpers as hp
import modtox.docking.glide.glide as dk
import modtox.data.pubchem as pchm
import modtox.data.dude as dd
import modtox.cpptraj.analisis as an
import modtox.docking.glide.analyse as gl
from sklearn.model_selection import train_test_split


folder = "/home/moruiz/cyp/new_test"
TRAIN_FOLDER = "from_train"
TEST_FOLDER = "from_test"
DOCKING_FOLDER = "docking"
ANALYSIS_FOLDER = "analysis"
DATASET_FOLDER = "dataset"
DESCRIPTORS_FOLDER = "descriptors"
METRICS_FOLDER = "metrics"


sdf_active_train = os.path.join(DATASET_FOLDER, "active_train.sdf")
sdf_inactive_train = os.path.join(DATASET_FOLDER, "inactive_train.sdf")
sdf_active_test = os.path.join(DATASET_FOLDER, "active_test.sdf")
sdf_inactive_test = os.path.join(DATASET_FOLDER, "inactive_test.sdf")

top = "/data/ModTox/5_CYPs/HROT_1r9o/1r9o_holo/1r9o_holo.top"
traj = "/data/ModTox/5_CYPs/HROT_1r9o/1r9o_holo/1R9O_*.x"
resname = "FLP"

clf='stack'
tpot=True
cv=10
mol_to_read = 100
dude = '/home/moruiz/cyp/dude/cp2c9'
pubchem = '/home/moruiz/cyp/pubchem'

def main(sdf_active_train, sdf_inactive_train, sdf_active_test, sdf_inactive_test, traj, resname, top, clf, tpot, cv, mol_to_read=None, RMSD=True, cluster=True, last=True, clust_type="BS", rmsd_type="BS", sieve=10, precision="SP", maxkeep=500, maxref=400, grid_mol=2, csv=False, best=False, glide_files="*dock_lib.maegz", database_train='dude', database_test='pubchem', dude=None, pubchem=None, set_prepare=False, debug=False):
    
    if not os.path.exists(TRAIN_FOLDER): os.mkdir(TRAIN_FOLDER)
    with hp.cd(TRAIN_FOLDER):
         
        if set_prepare:
            sdf_active_train, sdf_inactive_train, folder_to_get = set_preparation(traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type, sieve, database_train, mol_to_read, debug, train=True, test=False)

        csv_train = docking(sdf_active_train, sdf_inactive_train, precision, maxkeep, maxref, grid_mol, csv, glide_files, best, mol_to_read, debug, dock=False)
        model = build_model(sdf_active_train, sdf_inactive_train, csv_train, clf, tpot, cv, debug)
 
    if not os.path.exists(TEST_FOLDER): os.mkdir(TEST_FOLDER)
    with hp.cd(TEST_FOLDER):

        if set_prepare:
            sdf_active_test, sdf_inactive_test, _ = set_preparation(traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type, sieve, database_test, mol_to_read, debug, folder_to_get, train=False, test=True)
        csv_test = docking(sdf_active_test, sdf_inactive_test, precision, maxkeep, maxref, grid_mol, csv, glide_files, best, mol_to_read, debug, dock=False)
        predict_model(model, sdf_active_test, sdf_inactive_test, csv_test, clf, tpot, cv, debug)

def set_preparation(traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type, sieve, database, mol_to_read, debug, train, test, folder_to_get=None):

    if not os.path.exists(DATASET_FOLDER): os.mkdir(DATASET_FOLDER)
    
    #folder where used molecules during train are stored    
    if train: folder_to_get = os.path.abspath(DATASET_FOLDER)

    print("Extracting clusters from MD...")
    an.analise(ANALYSIS_FOLDER, traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type, sieve)

    print("Reading files....")
    if database == 'dude': sdf_active, sdf_inactive = dd.process_dude(dude, train, test, folder_output=DATASET_FOLDER, folder_to_get=folder_to_get, debug=debug)
    if database == 'pubchem': sdf_active, sdf_inactive = pchm.process_pubchem(pubchem, train, test, substrate, folder_output=DATASET_FOLDER, folder_to_get=folder_to_get, mol_to_read=mol_to_read, debug=debug)

    return sdf_active, sdf_inactive, folder_to_get


def docking(sdf_active, sdf_inactive, precision, maxkeep, maxref, grid_mol, csv, glide_files, best, mol_to_read, debug, dock):
    
    if not os.path.exists(DOCKING_FOLDER): os.mkdir(DOCKING_FOLDER)
    if not os.path.exists(DESCRIPTORS_FOLDER): os.mkdir(DESCRIPTORS_FOLDER)

    if dock: 
        print("Docking in process...")
        with hp.cd(DOCKING_FOLDER):
            docking_obj = dk.Glide_Docker(glob.glob(os.path.join(ANALYSIS_FOLDER, "*clust*.pdb")), [sdf_active, sdf_inactive], debug=debug)
            docking_obj.dock(precision=precision, maxkeep=maxkeep, maxref=maxref, grid_mol=grid_mol)

    print("Analyzing docking...")
    inp_files = glob.glob(os.path.join(DOCKING_FOLDER, glide_files))
    glide_csv = gl.analyze(inp_files, glide_dir=DESCRIPTORS_FOLDER, best=best, csv=csv, active=sdf_active, inactive=sdf_inactive, debug=debug)    
   
    return glide_csv

def build_model(sdf_active_train, sdf_inactive_train, csv_train, clf, tpot, cv, debug):
    
    #preprocess
    pre = Pre.ProcessorSDF(csv=csv_train, fp=False, descriptors=False, MACCS=False, columns=None)
    print("Fit and tranform for preprocessor..")
    X_train, y_train = pre.fit_transform(sdf_active=sdf_active_train, sdf_inactive=sdf_inactive_train, folder=DESCRIPTORS_FOLDER)
    print("Sanitazing...")
    X_train, y_train, mol_names = pre.sanitize(X_train, y_train)
    print("Filtering features...")
    pre.filter_features(X_train)
    
    #fit model
    Model = model.GenericModel(clf=clf, tpot=tpot,  cv=cv)
    print("Fitting model...")
    Model.fit(X_train,y_train)
    
    #postprocessing
    print("Postprocess for training ...")
    if not os.path.exists(METRICS_FOLDER): os.mkdir(METRICS_FOLDER)
    post = Post.PostProcessor(clf, Model.X_trans, Model.Y, Model.prediction_fit, Model.prediction_proba_fit, y_pred_test_clfs=Model.indiv_fit, folder=METRICS_FOLDER) 
    uncertainties = post.calculate_uncertanties()
    post.ROC()
    post.PR()
    post.conf_matrix()
    post.UMAP_plot()
    post.PCA_plot()
    post.tsne_plot()
   
    return Model
 
def predict_model(Model, sdf_active_test, sdf_inactive_test, csv_test, clf, tpot, cv, debug): 
    
    #preprocess test
    
    pre = Pre.ProcessorSDF(csv=csv_test, fp=False, descriptors=False, MACCS=False, columns=None)
    print("Fit and tranform for preprocessor..")
    X_test, y_test = pre.fit_transform(sdf_active=sdf_active_test, sdf_inactive=sdf_inactive_test, folder=DESCRIPTORS_FOLDER)
    print("Sanitazing...")
    X_test, y_test, mol_names = pre.sanitize(X_test, y_test)
    print("Filtering features...")
    pre.filter_features(X_test)
    
    #predict model
    print("Predicting...")
    y_pred = Model.predict(X_test, y_test, imputer=Model.imputer, scaler=Model.scaler, train_folder=os.path.join("../", TRAIN_FOLDER))
    
    #postprocessing
    print("Postprocess for test ...")
    if not os.path.exists(METRICS_FOLDER): os.mkdir(METRICS_FOLDER)
    post = Post.PostProcessor(clf, Model.X_test_trans, Model.Y_test, Model.prediction_test, Model.predictions_proba_test, y_pred_test_clfs=Model.indiv_pred, x_train=Model.X_trans, y_true_train=Model.Y, folder=METRICS_FOLDER) 
    uncertainties = post.calculate_uncertanties()
    post.ROC()
    post.PR()
    post.conf_matrix()
    post.UMAP_plot()
    post.PCA_plot()
    post.tsne_plot()
    post.shap_values(names=mol_names, features=pre.headers, debug=True)
    post.distributions(features=pre.headers, debug=True)
    post.feature_importance(features=pre.headers)
    post.domain_analysis(names=mol_names)


if __name__ == "__main__":

    main(sdf_active_train=sdf_active_train, sdf_inactive_train=sdf_inactive_train, sdf_active_test=sdf_active_test, sdf_inactive_test=sdf_inactive_test, traj=traj, resname=resname, top=top, clf=clf, tpot=tpot, cv=cv, dude=dude, pubchem=pubchem)

