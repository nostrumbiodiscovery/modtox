from modtox.modtox.new_models_classes.Features.feat_enum import Features
from rdkit import Chem
from typing import Dict, List

from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.DataStructs import FingerprintSimilarity
from modtox.modtox.new_models_classes._custom_errors import FeatureError

class BaseMolecule:
    """Base class for a molecule. Subclasses differ in init method."""
    molecule: Chem    
    features: Dict[Features, Dict[str, float or int]]
    similarities: List[float]

    def __init__(self) -> None:
        self.features = dict()
        self.similarities = list()
    
    def calculate_similarity(self, mol_list: List["Molecule"]):
        """Calculates the similarity to the provided list of molecules.
        Calculates the Tanimotos coefficient between self and every other
        molecule in the list."""
        
        for mol in mol_list:
            sim = FingerprintSimilarity(self.topo, mol.topo)
            self.similarities.append(sim)
        return self.similarity

    @property
    def has_glide(self):
        """Indicates if molecule has calculated Glide Features
        or all of them are set to 0."""
        if any(value != 0 for value in self.features[Features.glide].values()):
            return True
        else:
            return False

    @property
    def similarity(self):
        """Average of the similarities list."""
        return sum(self.similarities)/len(self.similarities)

    def add_feature(self, ft: Features, d: Dict[str, float or int]):
        """Adds a feature to the features dictionary. Provide a Feature
        and a dictionary as: {'feat1': 0, 'feat2': 1, ...}"""
        self.features[ft] = d

    def to_record(self, *features: Features):
        """Converts the molecule to a record format. If no *args are 
        supplied, includes all features."""
        
        if not features:
            features = [f for f in Features]
        
        record = {
            "Activity": self.activity,
            "is_external": self.is_external
        }
        
        for feature in features:
            if feature in self.features:
                record.update(self.features[feature])
            else:
                raise FeatureError(f"Feature {feature.name!r} has not been calculated for the collection.")

        return record

class MoleculeFromChem(BaseMolecule):
    """Molecule initialized from rdkit.Chem.Mol object with 
    '_Name' property containing activity information."""
    
    molecule: Chem
    scaffold: str
    name: str
    
    activity: bool
    is_external: bool
    
    features: Dict[Features, Dict[str, float or int]]
    similarities: List[float]

    def __init__(self, molecule) -> None:
        super().__init__()
        self.molecule = molecule
        self.scaffold = MurckoScaffoldSmiles(mol=self.molecule)
        self.name = molecule.GetProp("_Name")
        
        self.activity = "/active" in self.name
        self.is_external = False  # Default is False, extracting external set can set to True         

class MoleculeFromSMILES(BaseMolecule):
    """Molecule initialized from SMILES. 
    Supply activity and name."""
    molecule: Chem
    scaffold: str
    name: str or None
    
    activity: bool
    is_external: bool
    
    features: Dict[Features, Dict[str, float or int]]
    similarities: List[float]

    def __init__(self, smiles: str, activity: bool, name: str = None) -> None:
        super().__init__()
        self.molecule = Chem.MolFromSmiles(smiles)
        self.scaffold = MurckoScaffoldSmiles(mol=self.molecule)
        self.name = name

        self.activity = activity
        self.is_external = False  # Default is False, extracting external set can set to True   


# class UnknownMolecule(BaseMolecule):
#     """Molecule to classify after model is trained."""
#     def __init__(self, smiles, name=None) -> None:
#         self.molecule = Chem.MolFromSmiles(smiles)
#         self.name = name
#         self.scaffold = MurckoScaffoldSmiles(mol=self.molecule)        
#         self.features = dict()
#         self.similarities = list()
    
#     def classify(self):
#         raise NotImplementedError



