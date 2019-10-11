from rdkit import Chem
import numpy as np
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
from tqdm import tqdm
from rdkit.Chem import AllChem
import requests
try:
    from urllib import quote  # Python 2.X
except ImportError:
    from urllib.parse import quote  # Python 3+
import argparse
from argparse import RawTextHelpFormatter
import os
from chembl_webresource_client.unichem import unichem_client as unichem
from chembl_webresource_client.new_client import new_client
import modtox.Helpers.preprocess as pr


URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/{}.sdf"


class DUDE(object):
    
    def __init__(self, dude_folder, status):
        self.dude_folder = os.path.abspath(dude_folder)
        self.actives_ism = os.path.join(self.dude_folder, "actives_final.ism")
        self.decoys_ism = os.path.join(self.dude_folder, "decoys_final.ism")
        self.actives_sdf = os.path.join(self.dude_folder, "actives_final.sdf")
        self.decoys_sdf = os.path.join(self.dude_folder, "decoys_final.sdf")
        self.status = status
        
    def get_active_names(self):
        with open(self.actives_ism, "r") as f:
            self.active_names = [line.split()[-1] for line in f if line ]
            return self.active_names

    def get_inactive_names(self):
        with open(self.decoys_ism, "r") as f:
            self.inactive_names = [line.split()[-1] for line in f if line ]
            return self.inactive_names

    def retrieve_inchi_from_chembl(self, ids):
        for name in ids:
            for struct in unichem.structure(name,1):
                 yield str(struct["standardinchi"])


    def activities(self):
        pass 

    def to_inchi_key(self, chembl_names):
        for name in ids:
            for struct in unichem.structure(name,1):
                 yield str(struct["standardinchi"])

    def to_sdf(self, inchies, mol_names=None, output="active.sdf"):
        # Prepare output
        mol_names = mol_names if mol_names else range(len(inchies))
        outputfile = output.split('.')[0] + "_" + self.status + ".sdf"
        
        # Convert to sdf
        molecules_rdkit = [] ;w = Chem.SDWriter(outputfile)
        for inchy, name in tqdm(zip(inchies, mol_names)):
            try:
                m = Chem.inchi.MolFromInchi(inchy, removeHs=True)
                Chem.AssignStereochemistry(m)
                AllChem.EmbedMolecule(m)
                m.SetProp("_Name", name)
                m.SetProp("_MolFileChiralFlag", "1")
                molecules_rdkit.append(m)
            except IndexError:
                print("Molecules {} not found".format(name))
        return outputfile


    def filter_for_similarity(self, sdf_file, n_output_mols=100, output_sdf="inactive.sdf"):
        mols = np.array([m for m in Chem.SDMolSupplier(sdf_file, removeHs=False)])
        fps = [FingerprintMols.FingerprintMol(m) for m in mols]
        ref = fps[0]
        similarity = np.array([DataStructs.FingerprintSimilarity(ref, fp) for fp in fps[1:]])
        idx = np.round(np.linspace(0, len(similarity) - 1, n_output_mols)).astype(int)
        molecules_out = mols[idx]
        out = output_sdf.split('.')[0] + "_" + self.status + ".sdf"
        w = Chem.SDWriter(out) 
        for m in molecules_out: w.write(m)
        return out
        

def parse_args(parser):
    parser.add_argument("--dude",  type=str, help='DUD-E dataset folder')
    parser.add_argument("--output", type=str, help='sdf output', default="output.sdf")

def process_dude(dude_folder, status, output="cyp_actives.sdf", test=False, production=False):
    """
    Separate a dataset from dude into active/inactive
    having into account stereochemistry and tautomers
    """
    #If relative path move one down
    if os.path.abspath(dude_folder) == dude_folder:
        pass
    else:
        dude_folder = os.path.join("..", dude_folder)

    #If input zip is present decompress
    inputzip = os.path.join(dude_folder, "*.gz")
    if os.path.exists(inputzip):
        os.system("gunzip {}".format(inputzip))


    #Retrieve inchies
    dud_e = DUDE(dude_folder, status)
    active_names = dud_e.get_active_names()
    inchi_active = dud_e.retrieve_inchi_from_chembl(active_names)
    inactive_names = dud_e.get_inactive_names()
    inchi_inactive = dud_e.retrieve_inchi_from_chembl(inactive_names)
	
    #Retrieve active sdf
    dud_e.n_actives = len(active_names)
    active_output = dud_e.to_sdf(inchi_active, mol_names=active_names)
    if not test:
        active_output_proc = pr.ligprep(active_output)
    else:
        active_output_proc = active_output
    #Retrieve inactive sdf
    if production:
        #What will we do in production??
        pass
    else:
        inactive_output = dud_e.filter_for_similarity(dud_e.decoys_sdf, dud_e.n_actives)
    print("Files {}, {} created with chembl curated compounds".format(active_output_proc, inactive_output))
    return active_output_proc, inactive_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset from chembl by using ligprep over the actives having  \
    into account tautomerization, chirality and protonation of active ligands. Inactive are left as DUDE output them.", formatter_class=RawTextHelpFormatter)
    parse_args(parser)
    args = parser.parse_args()
    dude_set(args.dataset, args.output)
