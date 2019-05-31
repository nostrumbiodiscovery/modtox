import pandas as pd
import argparse
import numpy as np
import scipy as sc
import modtox.cpptraj.analisis as an


class Descriptor(an.CpptajBuilder):

    def __init__(self, traj_apo, top_apo, traj_holo, top_holo):
        self.apo = an.CpptajBuilder(traj_apo, top_apo)
        self.holo = an.CpptajBuilder(traj_holo, top_holo)

    def residue_motion(self):
        #Only takes protein atoms
        apo_residue_movement = self.apo.RMSD_byresidue(self.apo.traj)
        hol_residue_movement = self.holo.RMSD_byresidue(self.holo.traj)
        self.n_prot_residues = len(self.apo.rmsd_byres[:, 0])
        self.movement_diference = np.empty(self.n_prot_residues)
        for i in range(self.n_prot_residues):
            movement_apo_residue = np.mean(self.apo.rmsd_byres[i,:])
            movement_holo_residue = np.mean(self.holo.rmsd_byres[i,:])
            self.movement_diference[i] = movement_apo_residue-movement_holo_residue
        self.holo.plot_bars(self.movement_diference, output="rmsd_byres.png")
        self.holo.to_txt(self.movement_diference, output="rmsd_byres.txt")

    def residue_correlation(self):
        #res_num = self.holo.resnum("PLM")
        correlation =  self.holo.correlation_matrix(self.holo.traj, mask=':2')
        n_residues = len(self.holo.correlation[:, 0])
        self.correlation = np.empty(n_residues)
        for i in range(n_residues):
            self.correlation[i] = np.mean(self.holo.correlation[i,:])
        self.holo.plot_bars(self.correlation, output="correlation.png")
        self.holo.to_txt(self.correlation, output="correlation.txt")

    def residue_type(self):
        self.apo.residues(self.apo.traj)
        self.apo.to_txt(np.array(self.apo.sequence[0:self.n_prot_residues]), output="sequence.txt")
        #AMONIOACID TO CODE

    def residue_position(self):
        residues = range(1, self.n_prot_residues)
        frames = range(1, self.apo.traj.n_frames-1, 500)
        self.center_mass = np.array([np.mean(self.apo.compute_com(self.apo.traj, mask=':{}'.format(i), frame_indices=frames), axis=0) for i in residues])
        self.apo.to_txt(self.center_mass, output="com.txt")

def parse_args():
    parser = argparse.ArgumentParser(description='Specify trajectory and topology to be analised')
    parser.add_argument('trajapo', type=str, help='Trajectory apo to analise')
    parser.add_argument('trajholo', type=str, help='Trajectory holo to analise')
    parser.add_argument('--topapo', type=str, help='Topology apo of your trajectory')
    parser.add_argument('--topholo', type=str, help='Topology holo of your trajectory')
    args = parser.parse_args()
    return args.trajapo, args.topapo, args.trajholo, args.topholo

if __name__ == "__main__":
    traj_apo, top_apo, traj_holo, top_holo = parse_args()
    obj = Descriptor(traj_apo, top_apo, traj_holo, top_holo)
    obj.residue_motion()
    obj.residue_correlation()
    obj.residue_type()
    obj.residue_position()

    

