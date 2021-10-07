from modtox.modtox.new_models_classes.Features.add_feat import AddFeature, AddGlide, AddMordred, AddTopologicalFingerprints
from modtox.modtox.new_models_classes.mol import BaseMolecule, MoleculeFromChem
from modtox.modtox.new_models_classes.Features.feat_enum import Features
from modtox.modtox.new_models_classes._custom_errors import FeatureError

from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
import pandas as pd
import random
from typing import List
import os 
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter


# NO UNIT TESTS FOR COLLECTION
class BaseCollection:
    "Object gathering the features of collection of molecules."
    
    glide_features_csv: str
    molecules: List[BaseMolecule]
    
    def __init__(self) -> None:
        pass

    def add_features(self, *features: Features, glide_csv=None):
        if features is None:
            raise FeatureError("Must specify which features to calculate.")
        
        feat_map = {  # Maybe move this to feat_enum for consistency.
            Features.glide: AddGlide,
            Features.mordred: AddMordred,
            Features.topo: AddTopologicalFingerprints
        }

        for ft in features:
            feat_dict = feat_map[ft](self.molecules, glide_csv=glide_csv).calculate()  # glide_csv only needed for glide features, but others accept kwargs.
            for mol in self.molecules:
                mol.add_feature(ft, feat_dict[mol])
            
    def calculate_similarities(self):
        """Calculates similarities for all the molecules."""
        if Features.topo not in self.features_added:
            self.add_features(Features.topo)
        
        self.similarities = dict()
        for i, mol in enumerate(self.molecules):
            leave_one_out = self.molecules.copy()
            leave_one_out.pop(i)
            sim = mol.calculate_similarity(leave_one_out)
            self.similarities[mol] = sim     

    def balance(self, seed=None):
        # Rewrite method, maybe add a layer of abstraction for different balancing methods.
        """Balances the collection preferentially removing
        the molecules without glide features (all glide features == 0). """
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
        """Defines which molecules belong to the external set
        by seting the Molecule.is_external attribute to True (default = False).
        Also appends external molecules to a self.external_set"""
        num_mols = len(self.molecules)    # Total number of molecules
        ext_size = int(num_mols * proportion)  # Molecules to extract. 1/2 actives, 1/2 inactives
        if not ext_size % 2 == 0: ext_size -= 1
    
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
        """Converts collection to df to build the model.
        If no features are supplied, all features are added
        (see Molecule.to_record() method)."""
        records = list()
        for molecule in self.molecules:
            records.append(molecule.to_record(*features))
        df = pd.DataFrame(records)
        for col in df.columns: df[col] = pd.to_numeric(df[col], downcast="float", errors="coerce")
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

class CollectionFromSDF(BaseCollection):
    """Builds the collection from two SDF files."""
    def __init__(self, actives_sdf, inactives_sdf) -> None:
        self.read_sdf(actives_sdf, inactives_sdf)
        self.molecules = self.actives + self.inactives

        self.features_added = list()
        self.dataframes = list()

        self.summary = {"initial_molecules": len(self.molecules)}
    
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

class CollectionFromTarget(BaseCollection):
    """Builds a collection by target retrieving
    by target UniProt accession code."""
    pass

class CollectionSummarizer:
    """Class responsible for formatting output for user review."""
    def __init__(self, collection: BaseCollection, savedir=os.getcwd()) -> None:
        self.collection = collection
        self.savedir = savedir

    def plot_similarities(self, bins=15):
        actives_sim = [mol.similarity for mol in self.collection.molecules if mol.activity == True]
        inactives_sim = [mol.similarity for mol in self.collection.molecules if mol.activity == False]
        x = np.array([actives_sim, inactives_sim])
        plt.hist(x, bins=bins, stacked=True, density=True)
        plt.title("Collection similarity")
        plt.legend(["Actives", "Inactives"])
        plt.show()
        plt.savefig(os.path.join(self.savedir, "histogram_similarity.png"))
    
    def draw_most_representative_scaffolds(self, n):
        scaffolds = [mol.scaffold for mol in self.collection.molecules if mol.scaffold != ""]
        d = Counter(scaffolds).most_common(n)
        mols = [Chem.MolFromSmiles(smiles[0]) for smiles in d]
        leg = [f"Count: {count[1]}" for count in d]
        img = MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), legends=leg) 
        img.save(os.path.join(self.savedir, 'representative_scaffolds.png'))
        return

    def plot_scaffolds(self): 
        scaffolds = {mol: mol.scaffold for mol in self.collection.molecules if mol.scaffold != ""}
        Counter(scaffolds)
