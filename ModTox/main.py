import glob
import argparse
from argparse import RawTextHelpFormatter
from sklearn import svm
import ModTox.cpptraj.analisis as an
import ModTox.docking.glide.glide as dk
import ModTox.docking.glide.analyse as gl
import ModTox.ML.model as md

def parse_args():
    
    parser = argparse.ArgumentParser(description='Specify trajectories and topology to be analised.\n  \
    i.e python -m ModTox.main traj.xtc --top topology.pdb', formatter_class=RawTextHelpFormatter)
    an.parse_args(parser)
    gl.parse_args(parser)
    dk.parse_args(parser)
    md.parse_args(parser)
    parser.add_argument('--dock',  action="store_true", help='Topology of your trajectory')
    parser.add_argument('--analysis', action="store_true", help='Calculate RMSD plot')
    args = parser.parse_args()
    return args.traj, args.resname, args.active, args.inactive, args.top, args.glide_files, args.best, args.csv, args.RMSD, args.cluster, args.last, args.clust_type, args.rmsd_type, args.receptor, args.ligands_to_dock, args.grid, args.precision, args.maxkeep, args.maxref, args.dock, args.analysis, args.test, args.save, args.load, args.external_data, args.pb, args.cv

def main(traj, resname, active, inactive, top=None, glide_files="*dock*.maegz", best=False, csv=False, RMSD=True, cluster=True, last=True, clust_type="BS", rmsd_type="BS", receptor="*pv*.maegz", grid=None, precision="SP", maxkeep=500, maxref=400, dock=False, analysis=True, test=None, save=None, load=None, external_data=None, pb=False, cv=2):
    if dock:
        # Analyze trajectory&extract clusters
        an.analise(traj, resname, top, RMSD, cluster, last, clust_type, rmsd_type)
        # Cross dock all ligand to the extracted clusters
        docking_obj = dk.Glide_Docker(glob.glob("analisis/*clust*.pdb"), [active, inactive])
        docking_obj.dock(precision=precision, maxkeep=maxkeep, maxref=maxref)
        print("Docking in process... Once is finished run the same command exchanging --dock by --analysis flag to build model")
    elif analysis:
        # Analyze dockig files and build model features
        inp_files = glob.glob(glide_files)
        gl.analyze(inp_files, best=best, csv=csv, active=active, inactive=inactive)
        # Build Model
        clf = svm.SVC(C=1, gamma=1, kernel="linear")
        model = md.GenericModel(active, inactive, clf, csv=external_data, test=test)
        model.fit_transform(cv=cv, pb=pb)
        print("Model sucesfully build. Confusion_matrix.png outputted")



if __name__ == "__main__":
    trajs, resname, active, inactive, top, glide_files, best, csv, RMSD, cluster, last, clust_type, rmsd_type, \
       receptor, ligands_to_dock, grid, precision, maxkeep, maxref, dock, analysis, test, save, load, external_data, pb, cv = parse_args()
    main(trajs, resname, active, inactive, top, glide_files, best, csv, RMSD, cluster, last, clust_type, rmsd_type, 
        receptor, grid, precision, maxkeep, maxref, dock, analysis, test, save, load, external_data, pb, cv)
