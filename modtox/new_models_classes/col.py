from modtox.modtox.new_models_classes.mol import BaseMolecule, MoleculeFromChem
from modtox.modtox.new_models_classes.Features.feat_enum import Features

from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
import pandas as pd
import random
from typing import List
import os 
import matplotlib.pyplot as plt

from collections import Counter

class Collection:
    "Object gathering the features of collection of molecules."
    actives_sdf: str
    inactives_sdf: str
    glide_features_csv: str
    
    def __init__(self, actives_sdf, inactives_sdf) -> None:
        
        self.read_sdf(actives_sdf, inactives_sdf)
        self.molecules = self.actives + self.inactives

        self.summary = {"initial_molecules": len(self.molecules)}
        self.features_added = list()
        
        self.dataframes = list()

    def add_features(self, *features, glide_csv=None):
        pass
        # if not features:
        #     features = [ft for ft in Features]
        # features = [ft for ft in features if ft not in self.features_added]
        # AddFeatures(self, glide_csv, *features)
    
    def calculate_similarities(self):
        if Features.topo not in self.features_added:
            self.add_features(Features.topo)
        
        self.similarities = list()
        for i, mol in enumerate(self.molecules):
            leave_one_out = self.molecules.copy()
            leave_one_out.pop(i)
            sim = mol.calculate_similarity(leave_one_out)
            self.similarities.append(sim)
        return self.similarities

    def plot_similarities(self, savedir=os.getcwd(), bins=15):
        x = self.similarities
        plt.hist(x, bins=bins)
        plt.title("Collection similarity")
        plt.savefig(os.path.join(savedir, "histogram_similarity.png"))

    def representative_scaffolds(self, n, savedir=os.getcwd()):
        scaffolds = [mol.scaffold for mol in self.molecules if mol.scaffold != ""]
        d = Counter(scaffolds).most_common(n)

        mols = [Chem.MolFromSmiles(smiles[0]) for smiles in d]

        leg = [f"Count: {count[1]}" for count in d]
        img = MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), legends=leg) 
        img.save(os.path.join(savedir, 'representative_scaffolds.png'))
        return

    def balance(self, seed=None):
        self.waste = list()
        while len(self.actives) != len(self.inactives):
            if len(self.actives) > len(self.inactives):
                deleted_item = self.remove_random_molecule(self.actives, seed=seed) 
            elif len(self.actives) < len(self.inactives):
                deleted_item = self.remove_random_molecule(self.inactives, seed=seed)
            self.waste.append(deleted_item)
        self.molecules = self.actives + self.inactives
        return

    def extract_external_set(self, proportion, seed=None):
        num_mols = len(self.molecules)    # Total number of molecules
        ext_size = int(num_mols * proportion)  # Molecules to extract. 1/2 actives, 1/2 inactives
        if not ext_size % 2 == 0:
            ext_size = ext_size - 1
        
        random.seed(seed)  # For testing purposes

        self.external_set = list()
        for _ in range(int(ext_size/2)):
            random_active = random.choice(self.actives)
            while random_active.is_external == True:
                random_active = random.choice(self.actives)
            random_active.is_external = True
            self.external_set.append(random_active)
            
            random_inactive = random.choice(self.inactives)
            while random_inactive.is_external == True:
                random_inactive = random.choice(self.inactives)
            random_inactive.is_external = True
            self.external_set.append(random_inactive)
            
            self.molecules = self.actives + self.inactives
        return self.external_set
        
    def to_dataframe(self, *features):
        records = list()
        for molecule in self.molecules:
            records.append(molecule.to_record(*features))
        df = pd.DataFrame(records)
        df = self.format_df(df)
        return df

    @staticmethod
    def format_df(df: pd.DataFrame) -> None:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float", errors="coerce")
        df.dropna(axis="columns", how="all", inplace=True)
        df.drop(df.std()[(df.std() == 0)].index, axis=1, inplace=True)  # Drop columns where all values are equal (std=0)
        return df

    @staticmethod
    def remove_random_molecule(mol_list: List[BaseMolecule], seed=None):
        """Removes a random molecule from the list supplied, 
        preferably if doesn't have Glide Features"""
        random.seed(seed)
        pref_items = [mol for mol in mol_list if mol.has_glide == False]
        
        if pref_items:
            random_item = random.choice(pref_items)
        else:
            random_item = random.choice(mol_list)
        return mol_list.pop(mol_list.index(random_item))

    def read_sdf(self, actives_sdf, inactives_sdf):
        self.actives = [
            MoleculeFromChem(mol) 
            for mol in Chem.SDMolSupplier(actives_sdf)
            if ":" in mol.GetProp("_Name")
        ]
        self.inactives = [
            MoleculeFromChem(mol) 
            for mol in Chem.SDMolSupplier(inactives_sdf)
            if ":" in mol.GetProp("_Name")
        ]
    