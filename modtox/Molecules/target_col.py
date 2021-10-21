from dataclasses import dataclass
from typing import Dict, List

from modtox.modtox.Molecules.mol import MoleculeFromInChI
from modtox.modtox.Retrievers.bindingdb import RetrieveBDB
from modtox.modtox.Retrievers.chembl import RetrieveChEMBL
from modtox.modtox.Retrievers.pubchem import RetrievePubChem
from modtox.modtox.utils.enums import Database, Features, StandardTypes

from matplotlib_venn import venn3
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem

class IDs:
    def __init__(self) -> None:
        self.dbs = list()

class CollectionFromTarget:
    molecules: List[MoleculeFromInChI]
    def __init__(self, target) -> None:
        self.target = target
        self.molecules = list()

    def fetch(self, dbs: List[Database]):
        db_map = {
            Database.BindingDB: RetrieveBDB,
            Database.ChEMBL: RetrieveChEMBL,
            Database.PubChem: RetrievePubChem,
        }
        self.retrievers = {db: db_map.get(db)() for db in dbs}
        
        for ret in self.retrievers.values():
            ret.retrieve_by_target(self.target)

        self.activities = [
            activity
            for retriever in self.retrievers.values()
            for activity in retriever.activities
        ]

        self.ids = {db: ret.ids for db, ret in self.retrievers.items()}

        self.pubchem_activity = self.retrievers[Database.PubChem].tagged_activities

    def get_unique_inchis(self):
        self.unique_inchis = {
            inchi for id_dict in self.ids.values() for inchi in id_dict.keys()
        }

    def _unify_ids(self):
        self.unified_ids = {inchi: IDs() for inchi in self.unique_inchis}
        for db, ids_dict in self.ids.items():
            for inchi, id in ids_dict.items():
                self.unified_ids[inchi].dbs.append(db)
                setattr(self.unified_ids[inchi], f"{db.name.lower()}_id", id)

    def create_molecules(self):
        for i, inchi in enumerate(self.unique_inchis):
            mol = MoleculeFromInChI(inchi, f"{self.target}_{i}")
            
            mol.ids = self.unified_ids[inchi]
            
            for activity in [act for act in self.activities if act.inchi == inchi]:
                mol.add_activity(activity)
            
            self.molecules.append(mol)
    
    def assess_activity(self, criteria: Dict[StandardTypes, float or int]): # For pubchem, {"pubchem": 0}
        for std_type, std_val in criteria.items():
            if std_type == "pubchem":
                self.add_pubchem_activity()
            else:
                for molecule in self.molecules:
                    molecule.assess_activity(std_type, std_val)
    
    def add_pubchem_activity(self):
        for molecule in self.molecules:
            molecule.activity_tags["pubchem"] = self.pubchem_activity.get(molecule.inchi)

    def to_sdf(self, path):
        w = Chem.SDWriter()
        for mol in self.molecules:
            mol._add_properties_to_molecule()
            w.write(mol.molecule)

class RetrievalSummary:
    def __init__(self, collection: CollectionFromTarget) -> None:
        self.collection = collection

    def generate_molecules_venn_diagram(self):
        inchis = {db: set(ids_dict.keys()) for db, ids_dict in self.collection.ids.items()}
        venn3([s for s in inchis.values()], (db.name for db in inchis.keys()))
        plt.show()

            