import os
import numpy as np
import csv
import pubchempy as pcp
import argparse
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import modtox.Helpers.preprocess as pr


class PubChem():
     
    def __init__(self, pubchem_folder, stored_files, csv_filename, status, outputfile, substrate):
        if stored_files != False: self.unknown = False
        else: self.unknown = True
        self.csv_filename = csv_filename
        self.status = status
        self.folder = pubchem_folder
        self.outputfile = outputfile
        self.substrate = substrate
        self.splitting()

    def to_inchi(self, which_names):
           return [pcp.Compound.from_cid(int(i)).inchi for i in tqdm(which_names)]

    def splitting(self):
       if self.unknown:
           print('Reading from {}'.format(self.csv_filename))
           data = self.reading_from_pubchem()
       else:
           try: data = self.reading_from_file()
           except: 
               print('Need to provide a valid input file or to read the PubChem file')
       self.active_names = [mol for mol, activity in data.items() if activity == 'Active']
       self.inactive_names = [mol for mol, activity in data.items() if activity == 'Inactive']

    def to_sdf(self, actype):

        molecules = []
        molecules_rdkit = []
        output = actype + '_' + self.status +'.sdf'
        w = Chem.SDWriter(output)
        print('Filter and inchikey identification in process ... for {}'.format(actype))
        if actype == 'active': 
            iks, names = self.filtering(self.active_names)
            self.active_inchi = iks
            self.active_names = names
        
        if actype == 'inactive':
            iks, names = self.filtering(self.inactive_names)
            self.inactive_inchi = iks
            self.inactive_names = names

        for inchy, name in tqdm(zip(iks, names), total=len(names)-1):
            try:
                m = Chem.inchi.MolFromInchi(str(inchy))
                Chem.AssignStereochemistry(m)
                AllChem.EmbedMolecule(m)
                m.SetProp("_Name", name)
                m.SetProp("_MolFileChiralFlag", "1")
                molecules_rdkit.append(m)
            except IndexError:
                print("Molecules {} not found".format(name))
        
        for m in molecules_rdkit:             
            w.write(m)

        return output, len(names)


    def filtering(self, which_names):
        iks = self.to_inchi(which_names)
        #checking duplicates
        uniq = set(iks)
        if len(iks) > len(uniq): #cheking repeated inchikeys
            indices = { value : [ i for i, v in enumerate(iks) if v == value ] for value in uniq }
            filt_inchi = [iks[x[0]] for x in indices.values()] #filtered inchikeys
            filt_ids = [which_names[x[0]] for x in indices.values()] #filtering ids: we only get the first
            return filt_inchi, filt_ids
        else:
            print('Duplicates not detected') 
            return iks, which_names


    def reading_from_pubchem(self, total_molec = 100, trash_lines = 8):
        with open(os.path.join(self.folder, self.csv_filename), 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            idx = None; activities = {}; i = 0
            for row in spamreader:
                if i < total_molec + trash_lines:
                    try: idx = row.index(self.substrate + ' ' + 'Activity Outcome')        
                    except: pass
                    if idx != None and i > trash_lines: 
                        name = row[1]
                        activities[name] = row[idx]
                    i += 1
        with open(self.outputfile, 'wb') as op:
            pickle.dump(activities, op)
        return activities

    def reading_from_file(self): 
    
        with open(self.inputname, 'rb') as f:
            data = pickle.load(f)
        return data


def process_pubchem(pubchem_folder, csv_filename, status, substrate, stored_files = None, outputfile = 'inchi_all.pkl', test=False):
    pub_chem = PubChem(pubchem_folder, stored_files, csv_filename, status, outputfile, substrate)
    active_output, n_actives = pub_chem.to_sdf(actype = 'active')
    inactive_output, n_inactives = pub_chem.to_sdf(actype = 'inactive') 
    if not test: 
        output_proc = pr.ligprep(active_output)
    else:
        output_proc = active_output

    return output_proc, inactive_output

def parse_args(parser):
    parser.add_argument("--pubchem",  type=str, help='Pubchem folder')
    parser.add_argument("--stored_files",  action = 'store_true', help='Pubchem folder')
    parser.add_argument("--csv_filename", type=str, help = "csv filename with activities data (e.g. 'AID_1851_datatable_all.csv')")
    parser.add_argument("--substrate", type = str, help = "substrate name codification on csv file (e.g. p450-cyp2c9)") 
