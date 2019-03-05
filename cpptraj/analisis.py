import pytraj as pt
from matplotlib import pyplot as plt
import argparse
import glob
import numpy as np

class CpptajBuilder(object):

    def __init__(self, traj, topology):
        self.topology = topology if topology else traj[0]
        self.traj =  pt.iterload(traj, top=self.topology, autoimage=True) 

    def strip(self, ions=True, water=True, membrane=True, others=""):
        if ions:
            traj_nowat = pt.strip(self.traj, ":Na+, Cl-")
        if water:
            traj_nowat = pt.strip(traj_nowat, ":WAT")
        if membrane:
            traj_nowat = pt.strip(traj_nowat, ":Na+, Cl-")
        if others:
            traj_nowat = pt.strip(traj_nowat, others)
        pt.save('traj_strip_autoimaged.nc', traj_nowat, overwrite=True)
        pt.save('traj_strip_autoimaged.top', traj_nowat.top , overwrite=True)
        self.traj_strip = pt.iterload('traj_strip_autoimaged.nc', top='traj_strip_autoimaged.top', autoimage=True)

    def RMSD(self, traj, mask='', ref=0):
        self.rmsd_data = pt.rmsd(self.traj, mask=mask, ref=ref)
        print("RMSD done succesfully")

    def plot(self, values, output="rmsd.png"):
        assert isinstance(values, np.ndarray), "data rmsd must be a numpy array"
        frames = range(len(values))
        plt.plot(frames, values)
        plt.savefig(output)
        print("Plot done succesfully")


    def cluster(self, traj, mask='@CA', ref=0, options='sieve 5 summary clusters.out repout dani repfmt pdb'):
        self.clusters = pt.cluster.kmeans(traj,  n_clusters=10, mask=mask, options=options) 
        print("Clustering done succesfully")

def parse_args():
    
    parser = argparse.ArgumentParser(description='Specify trajectory and topology to be analised')
    parser.add_argument('traj', nargs="+", help='Trajectory to analise')
    parser.add_argument('--top', type=str, help='Topology of your trajectory')
    args = parser.parse_args()
    return args.traj, args.top

def analise(traj, top):
    trajectory = CpptajBuilder(traj, top)
    trajectory.strip()
    trajectory.RMSD(trajectory.traj_strip, mask="@CA")
    trajectory.plot(trajectory.rmsd_data)
    trajectory.cluster(trajectory.traj_strip)
    print("Trajectory {} sucessfuly analised".format(traj))


if __name__ == "__main__":
    traj, top = parse_args()
    analise(traj, top)
