import pytraj as pt
import sys
import os
from matplotlib import pyplot as plt
import argparse
import glob
import numpy as np

class CpptajBuilder(object):

    def __init__(self, traj, topology):
        self.topology = topology if topology else traj[0]
        self.traj =  pt.iterload(traj, top=self.topology, autoimage=True) 

    def strip(self, traj, ions=True, water=True, membrane=True, others="", output_path="analisis"):
	    # Check output path
	    if not os.path.exists(output_path):
	    	os.mkdir(output_path)
	    traj2 = pt.autoimage(traj[:])
            # Strip traj
            if ions:
                traj_nowat = pt.strip(traj2, ":Na+, Cl-")
            if water:
                traj_nowat = pt.strip(traj_nowat, ":WAT")
            if membrane:
                traj_nowat = pt.strip(traj_nowat, ":PA, PC, OL")
            if others:
                traj_nowat = pt.strip(traj_nowat, others)
            # Save output
	    n_frames = self.traj.n_frames
	    output_strip_top = os.path.join(output_path, 'traj_strip_autoimaged.top')
	    output_strip_converged_traj = os.path.join(output_path, 'traj_strip_converged.nc')
	    # Write strip traj
            pt.save(output_strip_top, traj_nowat.top , overwrite=True)
            pt.write_traj(output_strip_converged_traj, traj_nowat, overwrite=True, frame_indices=range(n_frames/2, n_frames-1))
	    # Load new traj
            self.traj_converged = pt.load(output_strip_converged_traj, top=output_strip_top)
 	    print("Preprocessed done succesfully")

    def RMSD(self, traj, mask='', ref=0):
        self.rmsd_data = pt.rmsd(traj, mask=mask, ref=ref)
        print("RMSD done succesfully")

    def RMSD_byresidue(self, traj, mask='', ref=0):
        self.rmsd_byres = np.array(pt.rmsd_perres(traj, mask=mask, ref=ref))
        print("RMSD done by res succesfully")

    def plot_line(self, values, output="rmsd.png", output_path="analisis"):
        # Check initial data
        assert isinstance(values, np.ndarray), "data rmsd must be a numpy array"
	    # Check output path
        if not os.path.exists(output_path):
	    	os.mkdir(output_path)
            # Plot
        frames = range(len(values))
        plt.plot(frames, values)
        plt.savefig(os.path.join(output_path, output))
        print("Line plot done succesfully")

    def plot_bars(self, values, output="rmsd.png", output_path="analisis"):
        # Check initial data
        assert isinstance(values, np.ndarray), "data rmsd must be a numpy array"
	    # Check output path
        if not os.path.exists(output_path):
	    	os.mkdir(output_path)
        # Plot
        frames = range(len(values))
        plt.bar(frames, values)
        plt.savefig(os.path.join(output_path, output))
        print("Bar plot done succesfully")

    def correlation_matrix(self, traj, mask="@CA"):
        self.correlation = np.array(pt.matrix.covar(traj, mask))
        return self.correlation


    def cluster(self, traj, mask='*', ref=0, options='sieve 10 normframe summary {0}/clusters.out repout {0}/cluster repfmt pdb',
        output_path="analisis"):
        # Check output path
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        # Cluster
        #output_cluster = os.path.join(output_path, "clusters.pdb")
        self.clusters = pt.cluster.hieragglo(traj, mask=mask, options=options.format(output_path), dtype='ndarray')
        #frames = [int(cluster) for cluster in self.clusters._cpp_out[1].split()[-10:]]
        #pt.write_traj(output_cluster, traj, overwrite=True, frame_indices=frames)
        print("Clustering done succesfully")

def parse_args():
    
    parser = argparse.ArgumentParser(description='Specify trajectory and topology to be analised')
    parser.add_argument('traj', nargs="+", help='Trajectory to analise')
    parser.add_argument('--top', type=str, help='Topology of your trajectory')
    args = parser.parse_args()
    return args.traj, args.top

def analise(traj, top):
    trajectory = CpptajBuilder(traj, top)
    trajectory.strip(trajectory.traj)
    trajectory.RMSD(trajectory.traj, mask="@CA")
    trajectory.plot_line(trajectory.rmsd_data)
    trajectory.cluster(trajectory.traj_converged)
    print("Trajectory {} sucessfuly analised".format(traj))


if __name__ == "__main__":
    traj, top = parse_args()
    analise(traj, top)
