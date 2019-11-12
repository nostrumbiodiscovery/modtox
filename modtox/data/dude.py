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
import modtox.Helpers.formats as ft


URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/{}.sdf"


class DUDE():
    
    def __init__(self, dude_folder, train, test, folder_output='.', folder_to_get = '.', output="cyp_actives.sdf", debug=False, production=False):

        self.dude_folder = os.path.abspath(dude_folder)
        #If relative path move one down
        if os.path.abspath(self.dude_folder) == self.dude_folder:
            pass
        else:
            self.dude_folder = os.path.join("..", self.dude_folder)
        self.actives_ism = os.path.join(self.dude_folder, "actives_final.ism")
        self.decoys_ism = os.path.join(self.dude_folder, "decoys_final.ism")
        self.actives_sdf = os.path.join(self.dude_folder, "actives_final.sdf")
        self.decoys_sdf = os.path.join(self.dude_folder, "decoys_final.sdf")
        self.used_mols = 'used_mols.txt'
        self.train = train
        self.test = test
        self.folder_to_get = folder_to_get
        self.output = output
        self.folder_output = folder_output
        self.debug = debug
        self.production = production


    def get_active_names(self):

        with open(self.actives_sdf, "r") as f:
            data = f.readlines()
            actives = [data[i+1].split('\n')[0] for i, line in enumerate(data[:-1]) if line == '$$$$\n']
            actives += [data[0].split('\n')[0]]
        with open(self.actives_ism, "r") as f:
            self.active_names = [line.split()[-1] for line in f if line ]
        assert len(self.active_names) == len(set(actives))
        return self.active_names

    def get_inactive_names(self):
        with open(self.decoys_sdf, "r") as f:
            data = f.readlines()
            inactives = [data[i+1].split('\n')[0] for i, line in enumerate(data[:-1]) if line == '$$$$\n']
            inactives += [data[0].split('\n')[0]]
        with open(self.decoys_ism, "r") as f:
            self.inactive_names = set([line.split()[-1] for line in f if line ])
        assert len(self.inactive_names) == len(set(inactives))
        return self.inactive_names

    def retrieve_inchi_from_chembl(self, ids):
        return [str(struct["standardinchi"]) for name in ids for struct in unichem.structure(name,1,2)]
    
    def retrieve_inchi_from_sdf(self, sdf):
        mols = Chem.SDMolSupplier(sdf)
        return [Chem.MolToInchi(mol) for mol in mols]

    def activities(self):
        pass 

    def to_sdf(self, inchies, mol_names=None, output="actives.sdf"):
        # Prepare output
        mol_names = mol_names if mol_names else range(len(inchies))
        outputfile = os.path.join(self.folder_output, output)
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

        for m in molecules_rdkit: w.write(m)
        
        return outputfile


    def filter_for_similarity(self, sdf_file, n_output_mols=100, output_sdf="inactives.sdf"):
        mols = np.array([m for m in Chem.SDMolSupplier(sdf_file, removeHs=False)])
        fps = [FingerprintMols.FingerprintMol(m) for m in mols]
        ref = fps[0]
        similarity = np.array([DataStructs.FingerprintSimilarity(ref, fp) for fp in fps[1:]])
        idx = np.round(np.linspace(0, len(similarity) - 1, n_output_mols)).astype(int)
        molecules_out = mols[idx]
        if not os.path.exists(self.folder_output): os.makedirs(self.folder_output)
        out = os.path.join(self.folder_output, output_sdf)
        w = Chem.SDWriter(out) 
        for m in molecules_out: w.write(m)
        return out

    def cleaning(self, inchi_active, active_names, inchi_inactive, inactive_names, folder_to_get):
                # recording instances from the training data
        
        if self.train:
            with open(os.path.join(self.folder_output, self.used_mols), 'w') as r: 
                for item in inchi_active + inchi_inactive:
                    r.write("{}\n".format(item))
       
        # extracting molecules from test already present in train
        if self.test:
            with open(os.path.join(folder_to_get, self.used_mols), 'r') as r:
                data = r.readlines()
                datalines = [x.split('\n')[0] for x in data]
                active_inchi_name = {inchi:name for inchi, name in zip(inchi_active, active_names)}
                inactive_inchi_name = {inchi:name for inchi, name in zip(inchi_inactive, inactive_names)} 

                inchi_active = [inchi for inchi in inchi_active if inchi not in datalines]
                inchi_inactive = [inchi for inchi in inchi_inactive if inchi not in datalines]
                
                #filtering by similarity >= 0.7
                mols_test_active = [Chem.inchi.MolFromInchi(str(inchi)) for inchi in inchi_active]
                mols_test_inactive = [Chem.inchi.MolFromInchi(str(inchi)) for inchi in inchi_inactive]
                mols_train = [Chem.inchi.MolFromInchi(str(inchi)) for inchi in datalines]
                fps_test_active = [Chem.Fingerprints.FingerprintMols.FingerprintMol(m) for m in mols_test_active]
                fps_test_inactive = [Chem.Fingerprints.FingerprintMols.FingerprintMol(m) for m in mols_test_inactive]
                fps_train = [Chem.Fingerprints.FingerprintMols.FingerprintMol(m) for m in mols_train]

                similarities_active = [rdkit.DataStructs.FingerprintSimilarity(fp_train, fp_test) for fp_test in fps_test_active for fp_train in fps_train]
                similarities_inactive = [rdkit.DataStructs.FingerprintSimilarity(fp_train, fp_test) for fp_test in fps_test_inactive for fp_train in fps_train]
                #all similarity-pairs
                to_keep = []
                for i in range(len(fps_test_active)):
                    sim_i = similarities_active[i*len(fps_train):(i+1)*len(fps_train)]
                    if len([s for s in sim_i if s<=0.7]) > 0 : to_keep.append(i)

                inchi_active = [inchi_active[i] for i in to_keep]

                to_keep = []
                for i in range(len(fps_test_inactive)):
                    sim_i = similarities_inactive[i*len(fps_train):(i+1)*len(fps_train)]
                    if len([s for s in sim_i if s<=0.7]) > 0 : to_keep.append(i)

                inchi_inactive = [inchi_inactive[i] for i in to_keep]
                
                #finally getting that names

                active_names = [active_inchi_name[inchi] for inchi in inchi_active]
                inactive_names = [inactive_inchi_name[inchi] for inchi in inchi_inactive]


        return inchi_active, active_names, inchi_inactive, inactive_names

    def process_dude(self):
        """
        Separate a dataset from dude into active/inactive
        having into account stereochemistry and tautomers
        """
        #If input zip is present decompress
        inputzip = os.path.join(self.dude_folder, "*.gz")
        if os.path.exists(inputzip):
            os.system("gunzip {}".format(inputzip))
        
        #Retrieve active inchies
        active_names = self.get_active_names()
    	
        self.n_actives = len(active_names)
        inchi_active = self.retrieve_inchi_from_chembl(active_names)   
        
        #Retrieve inactive sdf
        if self.production:
            #What will we do in production??
            pass
        else:
            inactive_output = self.filter_for_similarity(self.decoys_sdf, self.n_actives)
    
        #Retrieve inactive inchi
        inactive_names = self.get_inactive_names()

        inchi_inactive = self.retrieve_inchi_from_sdf(inactive_output)
    
        #Filter inchies
        if not self.production: 
            inchi_active, active_names, inchi_inactive, inactive_names = self.cleaning(inchi_active, active_names, inchi_inactive, inactive_names,  self.folder_to_get)
            print('Filter done')
        #Rewriting sdf for inactives
        inactive_output = self.to_sdf(inchi_inactive, mol_names = inactive_names, output = 'inactives.sdf')
    
        #sdf generation for actives
        active_output = self.to_sdf(inchi_active, mol_names=active_names)
        if not self.debug:
            active_output_proc = pr.ligprep(active_output, self.folder_output)
            active_output_proc = ft.mae_to_sd(active_output_proc, output=os.path.join(self.folder_output, 'actives.sdf'))
            
            inactive_output_proc = pr.ligprep(inactive_output, self.folder_output)
            inactive_output_proc = ft.mae_to_sd(inactive_output_proc, output=os.path.join(self.folder_output, 'inactives.sdf'))
        else:
            active_output_proc = active_output
            inactive_output_proc = inactive_output

        print("Files {}, {} created with chembl curated compounds".format(active_output_proc, inactive_output))
   
 
        print("Dude reading done!")
        return active_output_proc, inactive_output_proc

        
def parse_args(parser):
    parser.add_argument("--dude",  type=str, help='DUD-E dataset folder')
    parser.add_argument("--output", type=str, help='sdf output', default="output.sdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset from chembl by using ligprep over the actives having  \
    into account tautomerization, chirality and protonation of active ligands. Inactive are left as DUDE output them.", formatter_class=RawTextHelpFormatter)
    parse_args(parser)
    args = parser.parse_args()
    dude_set(args.dataset, args.output)
