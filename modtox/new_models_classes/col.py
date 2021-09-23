from mol import Molecule
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import random
from typing import Dict, List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np


class Features:
    name: str
    df = pd.DataFrame
    models: List

    def __init__(self, df: pd.DataFrame, glide=True, mordred=True, topo=True) -> None:
        self.name = 
        self.df = df
        self.models = list()
        return

    def to_Xy(self):
        y = self.df["Activity"]
        self.df.drop("Activity", axis=1, inplace=True)

        le = LabelEncoder()
        y = le.fit_transform(y)

        X = self.df
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        X = imputer.fit_transform(X)

        return X, y

class Collection:
    "Object gathering the features of collection of molecules."
    actives_sdf: str
    inactives_sdf: str
    glide_features_csv: str
    
    def __init__(self, actives_sdf, inactives_sdf) -> None:
        
        self.actives = [
            Molecule(mol) 
            for mol in Chem.SDMolSupplier(actives_sdf)
            if ":" in mol.GetProp("_Name")
        ]

        self.inactives = [
            Molecule(mol) 
            for mol in Chem.SDMolSupplier(inactives_sdf)
            if ":" in mol.GetProp("_Name")
        ]
        self.molecules = self.actives + self.inactives

        self.has_glide = False
        self.has_mordred = False
        self_has_topo = False

        self.dataframes = dict()

    def add_glide_features(self, glide_features_csv):
        self.has_glide = True
        df = pd.read_csv(glide_features_csv)
        # Drop columns before Title column
        cols_to_drop = [ df.columns[i] for i in range(df.columns.get_loc("Title")) ]
        cols_to_drop.append("Lig#")
        for col in cols_to_drop:
            df = df.drop(columns=col)
        df = df.set_index("Title")
        glide_d = df.to_dict("index")

        glide_columns = df.columns
        mol_names = [ mol.name for mol in self.molecules ]
        for (name, molecule) in zip(mol_names, self.molecules):
            if name in glide_d.keys():
                molecule.glide = glide_d[name]
                molecule.has_glide = True
            else:
                molecule.glide = dict.fromkeys(glide_columns, 0)
                molecule.has_glide = False
    
    def calculate_mordred(self):
        self.has_mordred = True
        print("Calculating mordred descriptors...")
        for molecule in tqdm(self.molecules):
            molecule.calculate_mordred()
    
    def calculate_topo(self):
        self.has_topo = True
        print("Calculating topological fingerprints...")
        for molecule in tqdm(self.molecules):
            molecule.calculate_topo()
     
    def balance(self):
        no_glide_act_mols = [ mol for mol in self.actives if mol.has_glide is False ]
        no_glide_inact_mols = [ mol for mol in self.inactives if mol.has_glide is False ]

        bal_summ = {
            "actives_init": len(self.actives),
            "unglid_act_init": len(no_glide_act_mols),
            "inactives_init": len(self.inactives),
            "unglid_inact_init": len(no_glide_inact_mols)
        }
        self.waste = list()
        while len(self.actives) != len(self.inactives):
            if len(self.actives) > len(self.inactives):
                del_item = self.remove_random_item(self.actives, no_glide_act_mols) 
            elif len(self.actives) < len(self.inactives):
                del_item = self.remove_random_item(self.inactives, no_glide_inact_mols)
            self.waste.append(del_item)
        
        bal_summ.update({
            "actives": len(self.actives),
            "unglid_act": len(no_glide_act_mols),
            "inactives": len(self.inactives),
            "unglid_inact": len(no_glide_inact_mols)
        }
        )
        self.molecules = self.actives + self.inactives
        return bal_summ

    def extract_external_set(self, proportion):
        num_mols = len(self.actives)    # Total number of molecules
        ext_size = num_mols * proportion  # Molecules to extract. 1/2 actives, 1/2 inactives
        if not ext_size % 2 == 0:
            ext_size = ext_size - 1
        
        self.external_set = list()
        for _ in range(ext_size/2):
            extracted_active = self.actives.pop(random.choice(self.actives))
            self.external_set.append(extracted_active)
            extracted_inactive = self.inactives.pop(random.choice(self.inactives))
            self.external_set.append(extracted_inactive)
        self.molecules = self.actives + self.inactives
        return self.external_set
        
    def to_dataframe(self, glide=True, mordred=True, topo=True):
        self.molecules = self.actives + self.inactives

        records = list()
        for molecule in self.molecules:
            records.append(molecule.to_record(glide=glide, mordred=mordred, topo=topo))
        df = pd.DataFrame(records)
        df = self.format_df(df)
        return Features(df, glide=glide, mordred=mordred, topo=topo)

    @staticmethod
    def format_df(df: pd.DataFrame) -> None:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float", errors="coerce")
        df.dropna(axis="columns", how="all", inplace=True)
        df.drop(df.std()[(df.std() == 0)].index, axis=1, inplace=True)  # Drop columns where all values are equal (std=0)
        return df

    @staticmethod
    def remove_random_item(lst: List, pref_items: List):
        """Removes a random item from a list preferably from indexes supplied."""
        if pref_items:
            random_item = pref_items.pop(random.choice(pref_items))
        else:
            random_item = random.choice(lst)

        return lst.pop(random_item)
        
