from abc import ABC, abstractmethod
from tqdm import tqdm
from rdkit import Chem
from mordred import Calculator, descriptors
import pandas as pd

from typing import List

from modtox.modtox.new_models_classes.Features.feat_enum import Features
from modtox.modtox.new_models_classes.mol import BaseMolecule, MoleculeFromChem

class AddFeature(ABC):
    """Base class for calculating/adding features."""
    def __init__(self, *molecules: BaseMolecule, **kwargs) -> None:
        super().__init__()
        self.molecules = molecules
    
    @abstractmethod
    def calculate(self):
        """Calculates the feature. Must be called from init method
        Must return a nested dict as: 
        { Molecule1: {'F1': 0, 'F2': 2.1, ...}, Molecule2: {'F1': 1.5, 'F2': 2.4, ...}, ...} 
        """

class AddGlide(AddFeature):
    """Reads 'glide_features.csv' and adds the features to the list of 
    supplied molecules. If a supplied molecule does not have a match in
    'glide_features.csv', all features are set to 0. 
    
    The molecule objects must be MoleculeFromChem because match between 
    molecule object and 'glide_features.csv' is done via the original 
    name in the SDF file: rdkit.Chem.Mol.GetProp('_Name').
    """
    def __init__(self, *molecules: MoleculeFromChem, glide_csv=None) -> None:  # Required MoleculeFromChem
        super().__init__(*molecules)
        self.glide_csv = glide_csv
        
    def calculate(self):
        df = self.format_glide(self.glide_csv)
        glide_d = df.to_dict("index")        
        features_names = df.columns
        mol_names = [mol.name for mol in self.molecules]
        
        features_dict = dict()
        for (name, mol) in zip(mol_names, self.molecules):  
            if name in glide_d.keys():
                features_dict[mol] = glide_d[name]
            else:
                features_dict[mol] = dict.fromkeys(features_names, 0)
        return features_dict
    
    @staticmethod
    def format_glide(csv_path):
        df = pd.read_csv(csv_path)
        # Drop columns before "Title"
        cols_to_drop = [ df.columns[i] for i in range(df.columns.get_loc("Title")) ]
        cols_to_drop.append("Lig#")
        for col in cols_to_drop:
            df = df.drop(columns=col)
        df = df.set_index("Title")
        return df

class AddTopologicalFingerprints(AddFeature):
    """Calculates the TopologicalFingerprints from rdkit"""
    def calculate(self):
        print("Calculating topological fingerprints...")
        features_dict = dict()
        for mol in tqdm(self.molecules):
            topo = Chem.RDKFingerprint(mol.molecule)
            mol.topo = topo  # For similarity calculation (stores rdkit obj to avoid calculating again.)
            d = {f"rdkit_fp_{i}": int(fp) for i, fp in enumerate(topo)}
            features_dict[mol] = d
        return features_dict

class AddMordred(AddFeature):
    """Calculates Mordred descriptors"""
    def calculate(self):
        print("Calculating mordred descriptors...")
        features_dict = dict()
        for mol in tqdm(self.molecules):
            mord = Calculator(descriptors, ignore_3D=True).pandas([mol.molecule], quiet=True)
            features_dict[mol] = mord.to_dict("index")[0]
        return features_dict



        
    


    
