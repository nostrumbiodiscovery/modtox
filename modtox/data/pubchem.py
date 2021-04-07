import os
import time
import pandas as pd
import itertools
import numpy as np
import csv
import pubchempy as pcp
import rdkit
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool
from rdkit.Chem import AllChem
import modtox.Helpers.preprocess as pr
import modtox.Helpers.formats as ft

class PubChem():
     
    def __init__(self, pubchem, train, test, substrate, folder_to_get='.', n_molecules_to_read=None, folder_output='.', production=False, debug=False):
        self.production = production
        self.csv_filename = os.path.abspath(pubchem)
        self.used_mols = 'used_mols.txt'
        self.test = test
        self.debug = debug
        self.production = production
        self.train = train
        self.folder_output = folder_output
        self.folder_to_get = folder_to_get
        self.substrate = substrate
        self.n_molecules_to_read = n_molecules_to_read
        self.splitting()

    def converting_to_inchi(self, name):
        rename = pcp.Compound.from_cid(int(name)).inchi
        time.sleep(0.6)
        return rename

    def to_inchi(self, which_names):
        cpus = 5
        pool = Pool(cpus)
        result = list(pool.imap(self.converting_to_inchi , tqdm(which_names)))
        pool.close()
        pool.join()
        
        return result

    def splitting(self):
       print('Reading from {}'.format(self.csv_filename))
       data = self.reading_from_pubchem()
       self.active_names = [mol for mol, activity in data.items() if activity == 'Active']
       self.inactive_names = [mol for mol, activity in data.items() if activity == 'Inactive']
       print('Discard of inconclusive essays done. Initial set: {} , Final set: {}'.format(len(data.items()), len(self.active_names) + len(self.inactive_names)))

    def to_sdf(self, readfromfile):

        molecules = []
        molecules_rdkit = []
        if self.train: where = "train"
        if self.test: where = "test"
        outputname = self.actype + '_' + where + '.sdf'
        if not os.path.exists(self.folder_output): os.mkdir(self.folder_output)
        self.output = os.path.join(self.folder_output, outputname)
        if os.path.exists(self.output): os.remove(self.output)
#        w = Chem.SDWriter(output)
        print('Filter and inchikey identification in process ... for {}'.format(self.actype))
        if self.actype == 'active':
            if not readfromfile:
                iks, names = self.filtering(self.active_names)
                self.active_inchi = iks
                self.active_names = names
                with open(os.path.join(self.folder_output, 'actives_info.csv'), 'w') as pp:
                    writer = csv.writer(pp)
                    writer.writerows(zip(iks, names))
            else:
                with open(os.path.join(self.folder_output, 'actives_info.csv'), 'r') as pp:
                    reader = csv.reader(pp)
                    names = []; iks = []
                    for row in reader:
                        iks.append(row[0])
                        names.append(row[1])
                self.active_inchi = iks
                self.active_names = names

        if self.actype == 'inactive':
            if not readfromfile:
                iks, names = self.filtering(self.inactive_names)
                self.inactive_inchi = iks
                self.inactive_names = names
                with open(os.path.join(self.folder_output, 'inactives_info.csv'), 'w') as pp:
                    writer = csv.writer(pp)
                    writer.writerows(zip(iks, names))
            else:
                with open(os.path.join(self.folder_output, 'inactives_info.csv'), 'r') as pp:
                    reader = csv.reader(pp)
                    names = []; iks = []
                    for row in reader:
                        iks.append(row[0])
                        names.append(row[1])
                self.inactive_inchi = iks
                self.inactive_names = names

            #removing from training repeated instances
            if not self.production: self.cleaning()
        print('Assigning stereochemistry...')
        cpus = 5
        pool = Pool(cpus)
        molecules_rdkit = list(tqdm(pool.imap(self.stereochem, itertools.zip_longest(iks, names)), total= len(names)))
        pool.close()
        pool.join()
        #removing molecules without stereochemical assignation
       # for mol in molecules_rdkit:
       #     w.write(mol)
        return self.output, len(names)
   
    def stereochem(self, inchy_name):
        wrongs = []
        m= None
        try:
            m = Chem.inchi.MolFromInchi(str(inchy_name[0]))
        except IndexError:
            print("Molecules {} not found".format(inchy_name[1]))
        try:
            Chem.AssignStereochemistry(m)
            AllChem.EmbedMolecule(m)
            name = str(inchy_name[1])
            m.SetProp("_Name", name)
            m.SetProp("_MolFileChiralFlag", "1")
            time.sleep(0.6)
        except: #to avoid ERROR: Sanitization error: Explicit valence for atom # 19 S, 8, is greater than permitted 
            pass
        if m == None:        
            if self.actype == "active":
                index = self.active_names.index(inchy_name[1])
                self.active_names = [name for i, inchi in enumerate(self.active_names) if i != index]
                self.active_inchi = [inchi for i, inchi in enumerate(self.active_inchi) if i != index]
            if self.actype == "inactive":
                index = self.inactive_names.index(inchy_name[1])
                self.inactive_inchi = [inchi for i, inchi in enumerate(self.inactive_inchi) if i != index]
                self.inactive_names = [inchi for i, inchi in enumerate(self.inactive_names) if i != index]
        else:
            outf=open(self.output,'a')
            writer = Chem.SDWriter(outf)
            writer.write(m)
        return m 

    def filtering(self, which_names):
        iks = self.to_inchi(which_names)
        #checking duplicates
        uniq = set(iks)
        if len(iks) > len(uniq): #cheking repeated inchikeys
            print('Duplicates detected')
            indices = { value : [ i for i, v in enumerate(iks) if v == value ] for value in uniq }
            iks = [iks[x[0]] for x in indices.values()] #filtered inchikeys
            which_names = [which_names[x[0]] for x in indices.values()] #filtering ids: we only get the first
        else:
            print('Duplicates not detected')

        return iks, which_names

    def cleaning(self):
                # recording instances from the training data
        if self.train:
            with open(os.path.join(self.folder_output, self.used_mols), 'w') as r:
                for item in self.active_inchi + self.inactive_inchi:
                    r.write("{}\n".format(item))

        # extracting molecules from test already present in train
        if self.test:
            with open(os.path.join(self.folder_to_get, self.used_mols), 'r') as r:
                data = r.readlines()
                datalines = [x.split('\n')[0] for x in data]
                #filtering by exact inchikey
                active_inchi = [inchi for inchi in self.active_inchi if inchi not in datalines]
                inactive_inchi = [inchi for inchi in self.inactive_inchi if inchi not in datalines]
                #filtering by similarity >= 0.7
                mols_test_active = [Chem.inchi.MolFromInchi(str(inchi)) for inchi in active_inchi]
                mols_test_inactive = [Chem.inchi.MolFromInchi(str(inchi)) for inchi in inactive_inchi]
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
                print('Active before cleaning: {}, after: {}'.format(len(active_inchi), len(to_keep)))
                self.active_inchi = [active_inchi[i] for i in to_keep]

                to_keep = []
                for i in range(len(fps_test_inactive)):
                    sim_i = similarities_inactive[i*len(fps_train):(i+1)*len(fps_train)]
                    if len([s for s in sim_i if s<=0.7]) > 0 : to_keep.append(i) 

                print('Inactive before cleaning: {}, after: {}'.format(len(inactive_inchi), len(to_keep)))

                self.inactive_inchi = [inactive_inchi[i] for i in to_keep]

        return 

    def reading_from_pubchem(self):

        data = pd.read_csv(self.csv_filename)
        cols = data.columns.values
        active_cols = [col for col in cols if 'Activity Outcome' in col]
        activity_labels = [data.loc[1,lab] for lab in active_cols]
        # split_labels = [ac.split() for ac in activity_labels]
        if self.substrate not in [ac.split()[0] for ac in activity_labels]:
            print('Not present in', activity_labels)
            return
        else:
            print('Label found')
            label = self.substrate + ' ' + 'Activity Outcome'
            useful_col = active_cols[activity_labels.index(label)]
            if self.n_molecules_to_read:
                end = self.n_molecules_to_read + 7
            else:
                end = len(data)
            useful_names = data['PUBCHEM_CID'].iloc[8:end, ]
            nonnamed = [i + 8 for i, name in enumerate(useful_names) if np.isnan(name)]
            idxs = list(range(8, end))
            idxs = [ids for ids in idxs if ids not in nonnamed]
            useful_names = data['PUBCHEM_CID'].iloc[idxs, ]
            useful_activities = data[useful_col].iloc[idxs, ]  # we add 8 to avoid the initial uninformative lines
            activities = {int(name): activity for name, activity in zip(useful_names, useful_activities)}
            return activities

    def process_pubchem(self, readfromfile=False):
        self.actype = 'active'
        active_output, n_actives = self.to_sdf(readfromfile)
        self.actype = 'inactive'
        inactive_output, n_inactives = self.to_sdf(readfromfile) 
        if not self.debug: 
            output_active_proc = pr.ligprep(active_output, output="active_processed.mae")
            output_active_proc = ft.mae_to_sd(output_active_proc, output=os.path.join(self.folder_output, 'actives.sdf'))
            output_inactive_proc = pr.ligprep(inactive_output, output="inactive_processed.mae")
            output_inactive_proc = ft.mae_to_sd(output_inactive_proc, output=os.path.join(self.folder_output, 'inactives.sdf'))
        else:
            output_active_proc = active_output
            output_inactive_proc = inactive_output
    
        return output_active_proc, output_inactive_proc

def parse_args(parser):
    parser.add_argument("--pubchem",  type=str, help='Pubchem file (e.g. AID_1851_datatable_all.csv)')
    parser.add_argument("--substrate", type = str, default = "p450-cyp2c9", help = "substrate name codification on csv file (e.g. p450-cyp2c9)") 
    parser.add_argument("--mol_to_read", type = int, help = "Number of molecules to read from pubchem (e.g. 100)") 
