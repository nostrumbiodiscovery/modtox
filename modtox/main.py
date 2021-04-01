import time
import sys
import glob
import os
import argparse
import numpy as np

import modtox.Helpers.preprocess as pr
import modtox.Helpers.formats as ft
import modtox.ML.preprocess as Pre
import modtox.ML.postprocess as Post
import modtox.ML.model2 as model
import modtox.Helpers.helpers as hp
import modtox.docking.glide.glide as dk
import modtox.docking.greasy.greasy_preparation as gre
import modtox.data.pubchem as pchm
import modtox.data.bindingdb as bdb
import modtox.data.dude as dd
import modtox.cpptraj.analisis as an
import modtox.docking.glide.analyse as gl


folder = "/home/moruiz/cyp/new_test"
TRAIN_FOLDER = "from_train"
TEST_FOLDER = "from_test"
DOCKING_FOLDER = "docking"
ANALYSIS_FOLDER = "analysis"
DATASET_FOLDER = "dataset"
DESCRIPTORS_FOLDER = "descriptors"
METRICS_FOLDER = "metrics"


def main(traj, resname, top, clf, tpot, cv, scoring='balanced_accuracy', mol_to_read=None, RMSD=True, cluster=True, last=True, clust_type="BS", rmsd_type="BS", sieve=10, precision="SP", maxkeep=500, maxref=400, grid_mol=2, csv=False, substrate=None, folder_to_get=None, best=False, glide_files="*pv.maegz", database_train='pubchem', database_test='bindingdb', dude=None, pubchem=None, binding=None, set_prepare=True, dock=True, build=True, predict=True, debug=False, greasy=True, sdf_active_train=None, sdf_inactive_train=None, sdf_active_test=None, sdf_inactive_test=None, csv_train=None, csv_test=None, majvoting=False, train=True, test=True, fp=False, descriptors=False, MACCS=False, columns=None, feature_to_check="external_descriptors", weighting=False, combine_model=False, ligprep=False):
   
    if sdf_active_train is not None: 
        sdf_active_train = os.path.abspath(sdf_active_train)
        sdf_inactive_train = os.path.abspath(sdf_inactive_train)
    if sdf_active_test is not None:
        sdf_active_test = os.path.abspath(sdf_active_test)
        sdf_inactive_test = os.path.abspath(sdf_inactive_test)
    if not os.path.exists(TRAIN_FOLDER): os.mkdir(TRAIN_FOLDER)
    if not os.path.exists(TEST_FOLDER): os.mkdir(TEST_FOLDER)

    ########################################## PREPARATION OF ACTIVE AND INACTIVE ##################################
    if set_prepare:
        if train:
            with hp.cd(TRAIN_FOLDER):
                sdf_active_train, sdf_inactive_train, folder_to_get = set_preparation(traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type, sieve, database_train, mol_to_read, substrate, debug, train=True, test=False)
        if test:
            with hp.cd(TEST_FOLDER):
                sdf_active_test, sdf_inactive_test, _ = set_preparation(traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type, sieve, database_test, mol_to_read, substrate, debug, train=False, test=True, folder_to_get=folder_to_get)
            

    ########################################################## DOCK ###########################################
    if dock:
        if train:
            with hp.cd(TRAIN_FOLDER):
                assert sdf_active_train !=None and sdf_inactive_train != None, "Sdf's must be provided! (use --sdf_active_train ,...)"
                docking(sdf_active_train, sdf_inactive_train, precision, maxkeep, maxref, grid_mol,mol_to_read, debug=False, greasy=greasy, ligprep=ligprep)
        if test:
            with hp.cd(TEST_FOLDER):
                assert sdf_active_test !=None and sdf_inactive_test != None, "sdf's must be provided! (use --sdf_active_test, ...)"
                docking(sdf_active_test, sdf_inactive_test, precision, maxkeep, maxref, grid_mol, mol_to_read, debug=False, greasy=greasy, ligprep=ligprep)

   ########################################################## GLIDE ANALYSIS  ###########################################
 
    if analysis:
        if train:
            with hp.cd(TRAIN_FOLDER):
                assert sdf_active_train !=None and sdf_inactive_train != None, "Sdf's must be provided! (use --sdf_active_train ,...)"
                csv_train = glide_analysis(glide_files, best, csv, sdf_active_train, sdf_inactive_train, debug, greasy)
        if test:
            with hp.cd(TEST_FOLDER):
                assert sdf_active_test !=None and sdf_inactive_test != None, "Sdf's must be provided! (use --sdf_active_test ,...)"
                csv_test = glide_analysis(glide_files, best, csv, sdf_active_test, sdf_inactive_test, debug, greasy)

    
    ########################################################## BUILD MODEL  ###########################################

    if build:
        if train:
            assert sdf_active_train !=None and sdf_inactive_train != None, "Sdf's must be provided! (use --sdf_active_train ,...)"
            if csv_train is not None: csv_train = os.path.abspath(csv_train)
            with hp.cd(TRAIN_FOLDER):
                model = build_model(sdf_active_train, sdf_inactive_train, csv_train, clf, tpot, scoring, cv, majvoting, fp, descriptors, MACCS, columns, feature_to_check, weighting, debug)
            train_folder = os.path.abspath(TRAIN_FOLDER)

    ########################################################## PREDICT  ###########################################

    if predict:
        if test:
            assert sdf_active_test !=None and sdf_inactive_test != None, "Sdf's must be provided! (use --sdf_active_train ,...)"
            if csv_test is not None: csv_test = os.path.abspath(csv_test)
            with hp.cd(TEST_FOLDER):
                y_pred  = predict_model(model, sdf_active_train, sdf_inactive_train, sdf_active_test, sdf_inactive_test, csv_test, clf, tpot, cv, majvoting, fp, descriptors, MACCS, columns, feature_to_check, train_folder, weighting, debug, postprocess=True)


    if combine_model:
        assert sdf_active_train !=None and sdf_inactive_train != None and sdf_active_train !=None and sdf_inactive_train != None, 'Provide sdfs please'
        combined_model(sdf_active_train, sdf_inactive_train, sdf_active_test, sdf_inactive_test, clf, tpot, scoring, cv, majvoting, columns, feature_to_check, weighting, debug, csv_train,  csv_test)
        

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
    if database == 'bindingdb': 
        DBase = bdb.BindingDB(binding, train, test, folder_output=DATASET_FOLDER, folder_to_get=folder_to_get, debug=debug)
        sdf_active, sdf_inactive = DBase.process_bind()

    return sdf_active, sdf_inactive, folder_to_get


def docking(sdf_active, sdf_inactive, precision, maxkeep, maxref, grid_mol, mol_to_read, debug, greasy, ligprep):
    
    if not os.path.exists(DOCKING_FOLDER): os.mkdir(DOCKING_FOLDER)
    if not os.path.exists(DESCRIPTORS_FOLDER): os.mkdir(DESCRIPTORS_FOLDER)

    if ligprep:
        print('Ligprep')
        output_active_proc = pr.ligprep(sdf_active, output="active_processed.mae")
        sdf_active = ft.mae_to_sd(output_active_proc, output=os.path.join(DATASET_FOLDER, 'actives.sdf'))
        output_inactive_proc = pr.ligprep(sdf_inactive, output="inactive_processed.mae")
        sdf_inactive = ft.mae_to_sd(output_inactive_proc, output=os.path.join(DATASET_FOLDER, 'inactives.sdf'))

    if greasy:
        print('Greasy preparation')
        folder = os.path.abspath(ANALYSIS_FOLDER)
        systems = glob.glob(os.path.join(DOCKING_FOLDER, "*grid.zip")) 
        if not len(systems) == 10:
            docking_obj = dk.Glide_Docker(glob.glob(os.path.join(folder, "*clust*.pdb")), [sdf_active, sdf_inactive], greasy=True, debug=debug)
            with hp.cd(DOCKING_FOLDER):
                docking_obj.dock(precision=precision, maxkeep=maxkeep, maxref=maxref, grid_mol=grid_mol)
        while not len(systems) == 10:
            systems = glob.glob(os.path.join(DOCKING_FOLDER, "*grid.zip")) 
            print(len(systems))
            time.sleep(5)

        greas = gre.GreasyObj(folder=DOCKING_FOLDER, active=sdf_active, inactive=sdf_inactive, systems= systems)
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
    return()

def glide_analysis(glide_files, best, csv, sdf_active, sdf_inactive, debug, greasy):

    print("Analyzing docking...")
    inp_files = glob.glob(os.path.join(DOCKING_FOLDER, glide_files))
    if greasy: 
        greas = gre.GreasyObj(folder=DOCKING_FOLDER, active=sdf_active, inactive=sdf_inactive, systems=[])
        inp_files = greas.merge_glides(inp_files)
    assert len(inp_files) == 10, "Use --greasy flag for merging files"
    glide_csv = gl.analyze(inp_files, glide_dir=DESCRIPTORS_FOLDER, best=best, csv=csv, active=sdf_active, inactive=sdf_inactive, debug=debug)    
   
    return glide_csv


def combined_model(sdf_active_train, sdf_inactive_train, sdf_active_test, sdf_inactive_test, clf, tpot, scoring, cv, majvoting, columns, feature_to_check, weighting, debug, csv_train, csv_test, mean=False):

    features_to_check = ['external_descriptors', 'fingerprintMACCS', 'descriptors', 'fingerprint' ]
    Combined = model.CombinedModel(csv_train, csv_test, features_to_check)

    for i in range(len(features_to_check)):

        bools_train, bools_test, feature_to_check = Combined.iteration(i)

        with hp.cd(TRAIN_FOLDER):
            print('BUILDING THE MODEL')
            Model = build_model(sdf_active_train, sdf_inactive_train, bools_train[0], clf, tpot, scoring, cv, majvoting, bools_train[3], bools_train[2], bools_train[1], columns, 
                                  feature_to_check, weighting, debug)
            mol_train = np.load('MOL_NAMES_train.npy')

        Y_TRUE_TRAIN = np.array(Model.Y)
        Y_PRED_TRAIN = np.array(Model.prediction_fit)

        train_folder = os.path.abspath(TRAIN_FOLDER)
        with hp.cd(TEST_FOLDER):
            print('PREDICTING')
            Y_PRED_TEST, Y_TRUE_TEST = predict_model(Model, sdf_active_train, sdf_inactive_train, sdf_active_test, sdf_inactive_test, bools_test[0], clf, tpot, cv, majvoting, bools_test[3], 
                                                       bools_test[2], bools_test[1], columns, feature_to_check, train_folder, weighting, debug, postprocess=False)
            mol_test = np.load('MOL_NAMES_test.npy')
        
        Combined.combined_prediction(i, mol_train, mol_test, Y_TRUE_TRAIN, Y_TRUE_TEST, Y_PRED_TRAIN, Y_PRED_TEST) 
    Combined.final_prediction()


def build_model(sdf_active_train, sdf_inactive_train, csv_train, clf, tpot, scoring, cv, majvoting, fp, descriptors, MACCS, columns, feature_to_check, weighting, debug):
  
    print('Building model with', sdf_active_train ,sdf_inactive_train)
    #preprocess
    pre = Pre.ProcessorSDF(csv=csv_train, fp=fp, descriptors=descriptors, MACCS=MACCS, columns=columns, label='train')
    #pre = Pre.ProcessorSDF(csv=False, fp=False, descriptors=False, MACCS=True, columns=None)
    print("Fit and tranform for preprocessor..")
    X_train, y_train = pre.fit_transform(sdf_active=sdf_active_train, sdf_inactive=sdf_inactive_train, folder=DESCRIPTORS_FOLDER)
    print("Sanitazing...")
    X_train, y_train, mol_names, y_removed, X_removed, cv = pre.sanitize(X_train, y_train, cv, folder=DESCRIPTORS_FOLDER, feature_to_check=feature_to_check)
    np.save("X_san", X_train)
    np.save("Y_san", y_train)
    print("Filtering features...")
    pre.filter_features(X_train)
    
    #fit model
    Model = model.GenericModel(X=X_train, Y=y_train, clf=clf, tpot=tpot,  scoring=scoring, cv=cv, X_removed=X_removed, y_removed=y_removed, majvoting=majvoting, weighting=weighting)
    print("Fitting model...")
    Model.fit(X_train,y_train)
    
    #postprocessing
    print("Postprocess for training ...")
    print(Model.X_trans.shape, Model.Y.shape)
    if not os.path.exists(METRICS_FOLDER): os.mkdir(METRICS_FOLDER)
    post = Post.PostProcessor(clf, Model.X_trans, Model.Y, Model.prediction_fit, Model.prediction_proba_fit, y_pred_test_clfs=Model.indiv_fit, folder=METRICS_FOLDER) 
    uncertainties = post.calculate_uncertanties()
    post.ROC()
    post.PR()
    post.conf_matrix()
   # post.UMAP_plot(single=True)
   # post.PCA_plot()
   # post.tsne_plot()
   
    return Model

def predict_model(Model, sdf_active_train, sdf_inactive_train, sdf_active_test, sdf_inactive_test, csv_test, clf, tpot, cv, majvoting, fp, descriptors, MACCS, columns, feature_to_check, train_folder, weighting, debug, postprocess): 
   
   
    print('Postprocess?', postprocess) 
    #preprocess test
    pre = Pre.ProcessorSDF(csv=csv_test, fp=fp, descriptors=descriptors, MACCS=MACCS, columns=columns, label='test')
    #pre = Pre.ProcessorSDF(csv=False, fp=False, descriptors=False, MACCS=True, columns=None)
    print("Fit and tranform for preprocessor..")
    X_test, y_test = pre.fit_transform(sdf_active=sdf_active_test, sdf_inactive=sdf_inactive_test, sdfs_to_compare=[sdf_active_train, sdf_inactive_train], folder=DESCRIPTORS_FOLDER)
    print("Sanitazing...")
    X_test, y_test, mol_names, y_removed, X_removed, cv = pre.sanitize(X_test, y_test, cv, folder=DESCRIPTORS_FOLDER, feature_to_check=feature_to_check)
    print("Filtering features...")
    pre.filter_features(X_test)
    
    #predict model
    print("Predicting...")
    y_pred, y_test = Model.predict(X_test, y_test, X_removed, y_removed, imputer=Model.imputer, scaler=Model.scaler, train_folder=train_folder)
    if postprocess: 
        #postprocessing
        print("Postprocess for test ...")
        print(Model.X_test_trans.shape, Model.Y_test.shape)
        if not os.path.exists(METRICS_FOLDER): os.mkdir(METRICS_FOLDER)
        if clf == 'stack':
            stack=True
            post = Post.PostProcessor(clf, Model.X_test_trans, Model.Y_test, Model.prediction_test, Model.predictions_proba_test, y_pred_test_clfs=Model.indiv_pred, x_train=Model.X_trans, y_true_train=Model.Y, y_pred_train_clfs=Model.proba_fit, y_proba_train=Model.prediction_proba_fit, folder=METRICS_FOLDER) 
        else: 
            stack=False
            post = Post.PostProcessor(clf, Model.X_test_trans, Model.Y_test, Model.prediction_test, Model.predictions_proba_test, y_pred_test_clfs=Model.indiv_pred, x_train=Model.X_trans, y_true_train=Model.Y, folder=METRICS_FOLDER)
        uncertainties = post.calculate_uncertanties()
        post.ROC()
        post.PR()
        post.conf_matrix()
        post.UMAP_plot(single=True, wrong=True, wrongall=True, traintest=True, wrongsingle=True, names=mol_names)
        post.PCA_plot()
        post.tsne_plot()
        post.shap_values(names=mol_names, features=pre.headers, debug=True)
        post.distributions(features=pre.headers, debug=False)
        post.feature_importance(features=pre.headers)
        post.domain_analysis(names=mol_names, stack=stack)
    
    return y_pred, y_test

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
    parser.add_argument('--mol_to_read',  default=None, type=int, help='Molecules to read from databases')
    parser.add_argument('--substrate',  default="p450-cyp2c9", type=str, help='Substrate name (only for pubchem)')
    parser.add_argument('--folder_to_get',  default=None, type=str, help='Folder to get used mols when running test')
    parser.add_argument('--dude',  default='/home/moruiz/cyp/dude/cp2c9', type=str, help='Path to dude files')
    parser.add_argument('--pubchem',  default='/home/moruiz/cyp/pubchem/AID_1851_datatable_all.csv', type=str, help='Pubchem file')
    parser.add_argument('--binding',  default='/home/moruiz/cyp/bindingdb/cyp2c9.sdf', type=str, help='Pubchem file')
    parser.add_argument('--sdf_active_train',  default=None, type=str, help='sdf file with actives for train')
    parser.add_argument('--sdf_inactive_train',  default=None, type=str, help='sdf file with inactives for train')
    parser.add_argument('--sdf_active_test',  default=None, type=str, help='sdf file with actives for test')
    parser.add_argument('--sdf_inactive_test',  default=None, type=str, help='sdf file with inactives for test')
    parser.add_argument('--csv_train',  default=None, type=str, help='glide csv file for train')
    parser.add_argument('--csv_test',  default=None, type=str, help='glide csv file for test')
    parser.add_argument('--database_train',  default='pubchem', type=str, help='database for train')
    parser.add_argument('--database_test',  default='bindingdb', type=str, help='database for test')
    parser.add_argument('--majvoting', action="store_true", help='Majority voting in stack model for last prediction')
    parser.add_argument('--train',  action="store_true", help='Run for training')
    parser.add_argument('--test',  action="store_true",  help='Run for testing')
    parser.add_argument('--fp',  action="store_true",  help='finger-print')
    parser.add_argument('--descriptors',  action="store_true",  help='descriptors')
    parser.add_argument('--MACCS',  action="store_true",  help='MACCS fingerprints')
    parser.add_argument('--columns',  default=None,  help='columns')
    parser.add_argument('--feature_to_check',  default="external_descriptors",  help='Feature to check nans')
    parser.add_argument('--scoring',  default="balanced_accuracy",  help='Scoring function for TPOT optimization')
    parser.add_argument('--weighting',  action="store_true",  help='Only for stack model. Weighting models')
    parser.add_argument('--combine_model',  action="store_true",  help='Only for stack model. Weighting models')
    parser.add_argument('--ligprep',  action="store_true",  help='Needed ligprep before docking')

    args = parser.parse_args()
    return args.set_prepare, args.dock, args.analysis, args.build, args.predict, args.greasy, args.top, args.traj, args.resname, args.clf, args.tpot, args.cv, args.mol_to_read, args.substrate, args.folder_to_get, args.dude, args.pubchem, args.binding, args.sdf_active_train, args.sdf_inactive_train,args.sdf_active_test, args.sdf_inactive_test, args.csv_train, args.csv_test, args.database_train, args.database_test, args.majvoting, args.train, args.test, args.fp, args.descriptors, args.MACCS, args.columns, args.feature_to_check, args.scoring, args.weighting, args.combine_model, args.ligprep


if __name__ == "__main__":

    set_prepare, dock, analysis, build, predict, greasy, top, traj, resname, clf, tpot, cv, mol_to_read, substrate, folder_to_get, dude, pubchem, binding, sdf_active_train, sdf_inactive_train, sdf_active_test, sdf_inactive_test, csv_train, csv_test, database_train, database_test, majvoting, train, test, fp, descriptors, MACCS, columns, feature_to_check, scoring, weighting, combine_model, ligprep = parse_args()
    main(traj=traj, resname=resname, top=top, clf=clf, tpot=tpot, cv=cv, scoring=scoring, dude=dude, pubchem=pubchem, binding=binding, greasy=greasy, set_prepare=set_prepare, dock=dock, build=build, predict=predict, mol_to_read=mol_to_read, substrate=substrate, folder_to_get=folder_to_get, sdf_active_train=sdf_active_train, sdf_inactive_train=sdf_inactive_train, sdf_active_test=sdf_active_test, sdf_inactive_test=sdf_inactive_test, csv_train=csv_train, csv_test=csv_test, database_train=database_train, database_test=database_test, majvoting=majvoting, train=train, test=test, fp=fp, descriptors=descriptors, MACCS=MACCS, columns=columns, feature_to_check=feature_to_check, weighting=weighting, combine_model=combine_model, ligprep=ligprep)
