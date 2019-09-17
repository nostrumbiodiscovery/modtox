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
import modtox.data.gpcrdb as gpcr

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
    gl.parse_args(parser)
    dk.parse_args(parser)
    md.parse_args(parser)
    parser.add_argument('--dock',  action="store_true", help='Topology of your trajectory')
    parser.add_argument('--analysis', action="store_true", help='Calculate RMSD plot')
    args = parser.parse_args()
    return args.traj, args.resname, args.active, args.inactive, args.top, args.glide_files, args.best, args.csv, args.RMSD, args.cluster, args.last, args.clust_type, args.rmsd_type, args.receptor, args.ligands_to_dock, args.grid, args.precision, args.maxkeep, args.maxref, args.dock, args.analysis, args.test, args.save, args.load, args.external_data, args.pb, args.cv, args.features, args.features_cv, args.descriptors, args.classifier, args.dude, args.grid_mol, args.clust_sieve

def main(traj, resname, active=None, inactive=None, top=None, glide_files="*dock*.maegz", best=False, csv=False, RMSD=True, cluster=True, last=True, clust_type="BS", rmsd_type="BS", receptor="*pv*.maegz", grid=None, precision="SP", maxkeep=500, maxref=400, dock=False, analysis=True, test=None, save=None, load=None, external_data=None, pb=False, cv=2, features=5, features_cv=1, descriptors=[], classifier="svm", dude=None, grid_mol=2, sieve=10, debug=False):
    if dock:
        # Analyze trajectory&extract clusters
        print("Extracting clusters from MD")
        if not os.path.exists("analisis"):
            an.analise(traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type, sieve)
        # Cross dock all ligand to the extracted clusters
        if dude:
            active, inactive = dd.process_dude(dude, test=debug)
        if active.split(".")[-1] == "csv":
            active = gpcr.process_gpcrdb(active)
            inactive = inactive
        print("Docking active and inactive dataset")
        if not debug:
            docking_obj = dk.Glide_Docker(glob.glob("analisis/*clust*.pdb"), [active, inactive], test=debug)
            docking_obj.dock(precision=precision, maxkeep=maxkeep, maxref=maxref, grid_mol=grid_mol)
            print("Docking in process... Once is finished run the same command exchanging --dock by --analysis flag to build model")
    elif analysis:
        if dude:
            active = "active.sdf"
            inactive = "inactive.sdf"
        # Analyze dockig files and build model features
        inp_files = glob.glob(glide_files)
        gl.analyze(inp_files, best=best, csv=csv, active=active, inactive=inactive)
        # Build Model
        for model in MODELS:
            try:
                model_obj = md.GenericModel(active, inactive, classifier, csv=model["csv"], test=test, pb=model["pb"], 
                    fp=model["fingerprint"], descriptors=model["descriptors"], MACCS=model["MACCS"])
                model_obj.build_model(cv=cv, output_conf=model["conf_matrix"])
                #model_obj.feature_importance(cl.XGBOOST, cv=features_cv, number_feat=features, output_features=model["output_feat"])
            except IOError:
                print("Model with descriptors not build for failure to connect to client webserver")
        print("Models sucesfully build. Confusion_matrix.png outputted")



if __name__ == "__main__":
    trajs, resname, active, inactive, top, glide_files, best, csv, RMSD, cluster, last, clust_type, rmsd_type, \
       receptor, ligands_to_dock, grid, precision, maxkeep, maxref, dock, analysis, test, \
       save, load, external_data, pb, cv, features, features_cv, descriptors, \
       classifier, dude, grid_mol, sieve = parse_args()
    main(trajs, resname, active, inactive, top, glide_files, best, csv, RMSD, cluster, last, clust_type, rmsd_type, 
        receptor, grid, precision, maxkeep, maxref, dock, analysis, test, save, load, external_data, pb, cv, features, features_cv, descriptors,
        classifier, dude, grid_mol, sieve)
