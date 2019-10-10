import matplotlib
matplotlib.use('Agg')
import modtox.ML.classifiers as cl
import modtox.cpptraj.analisis as an
import glob
import os
import argparse
import modtox.ML.model as md
from argparse import RawTextHelpFormatter
import modtox.docking.glide.glide as dk
import modtox.docking.glide.analyse as gl
import modtox.data.dude as dd
import modtox.Helpers.helpers as hp
import modtox.data.gpcrdb as gpcr
import modtox.data.pubchem as pchm

MODELS = [{"csv":None, "pb":True, "fingerprint":True, "MACCS":False, "descriptors":False,
            "output_feat":"fingerprint_important_features.txt", "conf_matrix":"fingerprint_conf_matrix.png"},
         {"csv":None, "pb":True, "fingerprint":False, "MACCS":True, "descriptors":False,
            "output_feat":"MACCS_important_features.txt", "conf_matrix":"MACCS_conf_matrix.png"},
         {"csv":None, "pb":True, "fingerprint":False, "MACCS":False, "descriptors":True,
            "output_feat":"descriptors_important_features.txt", "conf_matrix":"descriptors_conf_matrix.png"},
          {"csv":"glide_features.csv", "pb":False, "fingerprint":False, "MACCS":False, "descriptors":False,
            "output_feat":"glide_important_features.txt", "conf_matrix":"glide_conf_matrix.png"},
          {"csv":False, "pb":True, "fingerprint":True, "MACCS":True, "descriptors":False,
            "output_feat":"pb_important_features.txt", "conf_matrix":"pb_conf_matrix.png"}
         ]

MODELS = [          {"csv":"glide_features.csv", "pb":False, "fingerprint":False, "MACCS":False, "descriptors":False,
               "output_feat":"glide_important_features.txt", "conf_matrix":"glide_conf_matrix.png"}]

def parse_args():
    
    parser = argparse.ArgumentParser(description='Specify trajectories and topology to be analised.\n  \
    i.e python -m modtox.main traj.xtc --top topology.pdb', formatter_class=RawTextHelpFormatter,  conflict_handler='resolve')
    an.parse_args(parser)
    dd.parse_args(parser)
    pchm.parse_args(parser)
    gl.parse_args(parser)
    dk.parse_args(parser)
    md.parse_args(parser)
    parser.add_argument('--dock',  action="store_true", help='Topology of your trajectory')
    parser.add_argument('--assemble_model', action="store_true", help='Assemble model')
    parser.add_argument('--predict', action = 'store_true', help = 'Predict an external set')
    parser.add_argument('--debug', action="store_true", help='Run debug simulation')
    parser.add_argument('--output_dir', help='Folder to store modtox files', default="modtox_results")
    parser.add_argument('--status', default = "train", help= "Train or test")
    args = parser.parse_args()
    
    return args.traj, args.resname, args.active, args.inactive, args.top, args.glide_files, args.best, args.csv, args.RMSD, args.cluster, args.last, args.clust_type, args.rmsd_type, args.receptor, args.ligands_to_dock, args.grid, args.precision, args.maxkeep, args.maxref, args.dock, args.assemble_model, args.predict, args.test, args.save, args.load, args.external_data, args.pb, args.cv, args.features, args.features_cv, args.descriptors, args.classifier, args.filename_model, args.dude, args.pubchem, args.stored_files, args.csv_filename, args.substrate, args.grid_mol, args.clust_sieve, args.debug, args.output_dir, args.status, args.mol_to_read


def main(traj, resname, active=None, inactive=None, top=None, glide_files="*dock_lib.maegz", best=False, csv=False, RMSD=True, cluster=True, last=True, clust_type="BS", rmsd_type="BS", receptor="*pv*.maegz", grid=None, precision="SP", maxkeep=500, maxref=400, dock=False, assemble_model=True, predict = False, test=None, save=None, load=None, external_data=None, pb=False, cv=2, features=5, features_cv=1, descriptors=[], classifier="svm", filename_model = None, dude=None, pubchem = None, stored_files = False, csv_filename=None, substrate=None, grid_mol=2, sieve=10, debug=False, output_dir = "modtox_results", status = None, mol_to_read=None):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if dock:
        with hp.cd(output_dir):
            # Analyze trajectory&extract clusters
            print("Extracting clusters from MD")
            if not os.path.exists("analisis"):
                an.analise(traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type, sieve)
            # Cross dock all ligand to the extracted clusters
            if dude:
                active, inactive = dd.process_dude(dude, status, test=debug)
            if pubchem:
                active, inactive = pchm.process_pubchem(pubchem, stored_files = stored_files, csv_filename = csv_filename, status = status, substrate = substrate, mol_to_read=mol_to_read, test = debug)
            if active.split(".")[-1] == "csv":
                active = gpcr.process_gpcrdb(active)
                inactive = inactive
            print("Docking active and inactive dataset")
            if not debug:
                docking_obj = dk.Glide_Docker(glob.glob("analisis/*clust*.pdb"), [active, inactive], test=debug)
                docking_obj.dock(precision=precision, maxkeep=maxkeep, maxref=maxref, grid_mol=grid_mol)
                print("Docking in process... Once is finished run the same command exchanging --dock by --assemble_model flag to build model")
    if assemble_model:
        results_folder = 'from_train'
        if dude or pubchem:
            active = "active_train.sdf"
            inactive = "inactive_train.sdf"
        # Analyze dockig files and build model features
        inp_files = glob.glob(glide_files)
        gl.analyze(inp_files, best=best, csv=csv, active=active, inactive=inactive, debug=debug)
        # Build Model
        for model in MODELS:
            try:
                model_obj = md.GenericModel(active, inactive, classifier, True, filename_model, results_folder = results_folder, csv=model["csv"], test=test, pb=model["pb"], fp=model["fingerprint"], descriptors=model["descriptors"], MACCS=model["MACCS"])
                model_obj.build_model(cv=cv, output_conf=model["conf_matrix"])
                #model_obj.feature_importance(cl.XGBOOST, cv=features_cv, number_feat=features, output_features=model["output_feat"])
            except IOError as e:
                print(e)
                print("Model with descriptors not build for failure to connect to client webserver")
        print("Models sucesfully build. Confusion_matrix.png outputted")
 
    if predict:
        results_folder = 'from_test'
        assert os.path.isfile(filename_model)== True, "Run the training of the module first with flag --build_model. More in docs"
        if dude or pubchem:
            active = "active_test.sdf"
            inactive = "inactive_test.sdf"

        inp_files = glob.glob(glide_files)
        gl.analyze(inp_files, best=best, csv=csv, active=active, inactive=inactive, debug=debug)
        for model in MODELS: 
            model_obj = md.GenericModel(active, inactive, classifier, True, filename_model, results_folder = results_folder, csv=model["csv"], test=test, pb=model["pb"], fp=model["fingerprint"], descriptors=model["descriptors"], MACCS=model["MACCS"])
            model_obj.external_prediction()

if __name__ == "__main__":
    trajs, resname, active, inactive, top, glide_files, best, csv, RMSD, cluster, last, clust_type, rmsd_type, \
    receptor, ligands_to_dock, grid, precision, maxkeep, maxref, dock, assemble_model, predict, test, \
    save, load, external_data, pb, cv, features, features_cv, descriptors, \
    classifier, filename_model, dude, pubchem, stored_files, csv_filename, substrate, grid_mol, sieve, debug, output_dir, status, mol_to_read = parse_args()
    main(trajs, resname, active, inactive, top, glide_files, best, csv, RMSD, cluster, last, clust_type, rmsd_type, 
        receptor, grid, precision, maxkeep, maxref, dock, assemble_model, predict, test, save, load, external_data, pb, cv, features, features_cv, 
        descriptors, classifier, filename_model, dude, pubchem, stored_files, csv_filename, substrate, grid_mol, sieve, debug, output_dir, status, mol_to_read)
