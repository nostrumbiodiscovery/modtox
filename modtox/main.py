import sys
import glob
import os
import argparse
import subprocess
import numpy as np
import modtox.ML.preprocess as Pre
import modtox.ML.postprocess as Post
import modtox.ML.model2 as model
import modtox.Helpers.helpers as hp
import modtox.docking.glide.glide as dk
import modtox.docking.greasy.greasy_preparation as gre
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


def main(traj, resname, top, clf, tpot, cv, mol_to_read=None, RMSD=True, cluster=True, last=True, clust_type="BS", rmsd_type="BS", sieve=10, precision="SP", maxkeep=500, maxref=400, grid_mol=2, csv=False, substrate=None, best=False, glide_files="*maegz", database_train='dude', database_test='pubchem', dude=None, pubchem=None, set_prepare=True, dock=True, build=True, predict=True, debug=False, greasy=True):
    
    if not os.path.exists(TRAIN_FOLDER): os.mkdir(TRAIN_FOLDER)
    if not os.path.exists(TEST_FOLDER): os.mkdir(TEST_FOLDER)

    ########################################## PREPARATION OF ACTIVE AND INACTIVE ##################################
    if set_prepare:
        with hp.cd(TRAIN_FOLDER):
            sdf_active_train, sdf_inactive_train, folder_to_get = set_preparation(traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type, sieve, database_train, mol_to_read, substrate, debug, train=True, test=False)

        with hp.cd(TEST_FOLDER):
            sdf_active_test, sdf_inactive_test, _ = set_preparation(traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type, sieve, database_test, mol_to_read, substrate, debug, train=False, test=True, folder_to_get=folder_to_get)
        
    ########################################################## DOCK ###########################################
    if dock:

        with hp.cd(TRAIN_FOLDER):
  
            sdf_active_train = "dataset/actives.sdf"
            sdf_inactive_train = "dataset/inactives.sdf"
  
            docking(sdf_active_train, sdf_inactive_train, precision, maxkeep, maxref, grid_mol,mol_to_read, debug=True, greasy=greasy)
        with hp.cd(TEST_FOLDER):

    #        sdf_active_test = "dataset/actives.sdf"
    #        sdf_inactive_test = "dataset/inactives.sdf"

            docking(sdf_active_test, sdf_inactive_test, precision, maxkeep, maxref, grid_mol, mol_to_read, debug=True, greasy=greasy)

   ########################################################## GLIDE ANALYSIS  ###########################################
 
    if analysis:
        sdf_active_train = "dataset/actives.sdf"
        sdf_inactive_train = "dataset/inactives.sdf"
        with hp.cd(TRAIN_FOLDER):
            csv_train = glide_analysis(glide_files, best, csv, sdf_active_train, sdf_inactive_train, debug)
 #       with hp.cd(TEST_FOLDER):
 #           csv_test = glide_analysis(glide_files, best, csv, sdf_active_test, sdf_inactive_test, debug)

    
    ########################################################## BUILD MODEL  ###########################################

    if build:
        with hp.cd(TRAIN_FOLDER):
            model = build_model(sdf_active_train, sdf_inactive_train, csv_train, clf, tpot, cv, debug)

    ########################################################## PREDICT  ###########################################

    if predict:
        with hp.cd(TEST_FOLDER):
            predict_model(model, sdf_active_test, sdf_inactive_test, csv_test, clf, tpot, cv, debug)


def set_preparation(traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type, sieve, database, mol_to_read, substrate, debug, train, test, folder_to_get=None):

    if not os.path.exists(DATASET_FOLDER): os.mkdir(DATASET_FOLDER)
    
    #folder where used molecules during train are stored    
    if train: folder_to_get = os.path.abspath(DATASET_FOLDER)
    print("Extracting clusters from MD...")
    an.analise(traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type, sieve, output_dir=ANALYSIS_FOLDER)
    print("Reading files....")
    if database == 'pubchem': 
        DBase = pchm.PubChem(pubchem, train, test,substrate, folder_output=DATASET_FOLDER, n_molecules_to_read=mol_to_read, folder_to_get=folder_to_get, production=False, debug=debug)
        sdf_active, sdf_inactive = DBase.process_pubchem()
    if database == 'dude': 
        DBase = dd.DUDE(dude, train, test, folder_output=DATASET_FOLDER, folder_to_get=folder_to_get, debug=debug)
        sdf_active, sdf_inactive = DBase.process_dude()

    return sdf_active, sdf_inactive, folder_to_get


def docking(sdf_active, sdf_inactive, precision, maxkeep, maxref, grid_mol, mol_to_read, debug, greasy):
    
    if not os.path.exists(DOCKING_FOLDER): os.mkdir(DOCKING_FOLDER)
    if not os.path.exists(DESCRIPTORS_FOLDER): os.mkdir(DESCRIPTORS_FOLDER)

    if greasy:
        print('Greasy preparation')
        folder = os.path.abspath(ANALYSIS_FOLDER)
        sdf_active = os.path.abspath(sdf_active)
        sdf_inactive = os.path.abspath(sdf_inactive)
        docking_obj = dk.Glide_Docker(glob.glob(os.path.join(folder, "*clust*.pdb")), [sdf_active, sdf_inactive], greasy=True, debug=debug)
        with hp.cd(DOCKING_FOLDER):
                docking_obj.dock(precision=precision, maxkeep=maxkeep, maxref=maxref, grid_mol=grid_mol)
        greas = gre.GreasyObj(folder=DOCKING_FOLDER, active=sdf_active, inactive=sdf_inactive, systems=glob.glob(os.path.join(DOCKING_FOLDER, "*grid.zip")))
        greas.preparation()
        sys.exit('Waiting for greasy results')
    else:
        print("Docking in process...")
        folder = os.path.abspath(ANALYSIS_FOLDER)
        sdf_active = os.path.abspath(sdf_active)
        sdf_inactive = os.path.abspath(sdf_inactive)
        docking_obj = dk.Glide_Docker(glob.glob(os.path.join(folder, "*clust*.pdb")), [sdf_active, sdf_inactive], debug=debug)
        with hp.cd(DOCKING_FOLDER):
                docking_obj.dock(precision=precision, maxkeep=maxkeep, maxref=maxref, grid_mol=grid_mol)
                if debug == False: sys.exit('Waiting for docking results')
    return

def glide_analysis(glide_files, best, csv, sdf_active, sdf_inactive, debug, greasy):

    print("Analyzing docking...")

    print("Analyzing docking...")
    inp_files = glob.glob(os.path.join(DOCKING_FOLDER, glide_files))
    if greasy: 
        greas = gre.GreasyObj(folder=DOCKING_FOLDER, active=sdf_active, inactive=sdf_inactive, systems=[])
        inp_files = greas.merge_glides(inp_files)
    glide_csv = gl.analyze(inp_files, glide_dir=DESCRIPTORS_FOLDER, best=best, csv=csv, active=sdf_active, inactive=sdf_inactive, debug=debug)    
   
    return glide_csv



def build_model(sdf_active_train, sdf_inactive_train, csv_train, clf, tpot, cv, debug):
    

    import pdb; pdb.set_trace()
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
    post.UMAP_plot(single=True)
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
    post.UMAP_plot(single=True, wrong=True, wrongall=True, traintest=True, wrongsingle=True)
    post.PCA_plot()
    post.tsne_plot()
    post.shap_values(names=mol_names, features=pre.headers, debug=True)
    post.distributions(features=pre.headers, debug=True)
    post.feature_importance(features=pre.headers)
    post.domain_analysis(names=mol_names)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--set_prepare',  action="store_true", help='Preparation of files from MD')
    parser.add_argument('--dock',  action="store_true", help='Flag for docking')
    parser.add_argument('--analysis',  action="store_true", help='Extract csv_file from glide results')
    parser.add_argument('--build',  action="store_true", help='Build ML model')
    parser.add_argument('--predict',  action="store_true", help='Predict from ML loaded model')
    parser.add_argument('--greasy',  action="store_true", help='Prepare files to dock with greasy')
    parser.add_argument('--top',  default="/data/ModTox/5_CYPs/HROT_1r9o/1r9o_holo/1r9o_holo.top", help='Topology')
    parser.add_argument('--traj',  default="/data/ModTox/5_CYPs/HROT_1r9o/1r9o_holo/1R9O_*.x", help='Trajectory files')
    parser.add_argument('--resname',  default="FLP", help='Residue name')
    parser.add_argument('--clf',  default='single', help='Classifier type: single/stack')
    parser.add_argument('--tpot',  action="store_true", help='Use TPOT to build model')
    parser.add_argument('--cv',  default=10, type=int, help='Cross-validation')
    parser.add_argument('--mol_to_read',  default=100, type=int, help='Molecules to read from databases')
    parser.add_argument('--substrate',  default="p450-cyp2c9", type=str, help='Substrate name (only for pubchem)')
    parser.add_argument('--dude',  default='/home/moruiz/cyp/dude/cp2c9', type=str, help='Path to dude files')
    parser.add_argument('--pubchem',  default='/home/moruiz/cyp/pubchem/AID_1851_datatable_all.csv', type=str, help='Pubchem file')
    args = parser.parse_args()
    return args.set_prepare, args.dock, args.analysis, args.build, args.predict, args.greasy, args.top, args.traj, args.resname, args.clf, args.tpot, args.cv, args.mol_to_read, args.substrate, args.dude, args.pubchem


if __name__ == "__main__":

    set_prepare, dock, analysis, build, predict, greasy, top, traj, resname, clf, tpot, cv, mol_to_read, substrate, dude, pubchem = parse_args()
    main(traj=traj, resname=resname, top=top, clf=clf, tpot=tpot, cv=cv, dude=dude, pubchem=pubchem, greasy=greasy, set_prepare=set_prepare, dock=dock, build=build, predict=predict, mol_to_read=mol_to_read, substrate=substrate)

