from modtox.modtox.utils.enums import Features
from rdkit import Chem
from typing import Collection, Dict, List
from dataclasses import dataclass
from modtox.modtox.utils._custom_errors import FeatureError

@dataclass
class Cluster:
    id: int
    members: int
    is_centroid: bool

class BaseMolecule:
    """Represents the base molecule. Should not instantiated."""
    molecule: Chem
    features: Dict[Features, Dict[str, float or int]]
    scaffold: str

    def __init__(self) -> None:
        """Creates empty feature dictionary to be populated as
        Features.xxx: {"F1": 1, "F2": 0.5, ...}, Features.yyy: {...}.
        Creates empty similarity list. 
        """
        self.features = dict()

    @property
    def has_glide(self):
        """Checks if molecule has original Glide features or have all
        been set to 0.

        Returns
        -------
        Bool
            True for original features, False for all features set to 0.
        """
        if any(value != 0 for value in self.features[Features.glide].values()):
            return True
        else:
            return False

    def add_cluster_info(self, cluster_id: int, members: int, is_centroid: bool):
        self.cluster = Cluster(cluster_id, members, is_centroid)

    def add_feature(self, ft: Features, d: Dict[str, float or int]):
        """Adds a feature to the features dictionary.

        Parameters
        ----------
        ft : Features
            Instance of the Enum Features class to be added.
        d : Dict[str, float or int]
            Formatted as as: {'F1': 0, 'F2': 1.1, ...}"""

        self.features[ft] = d
        
    def to_record(self, *features: Features):
        """Converts the molecule to a record format. Contains the activity,
        is_external and the specified features.If no *args are supplied, 
        includes all features.

        Parameters
        ----------
        *features : Features
            Instance(s) of the Enum Features class to be added.
        
        Returns
        -------
        Dict
            Dictionary with 

        Raises
        ------
        FeatureError
            Error raised when the specified Feature is not in the
            self.features or is not even a Features(Enum) instance. 
        """

        if not features:
            features = [f for f in Features]

        record = {
            "Activity": self.activity,
            "cluster_id": self.cluster.id,
            "cluster_members": self.cluster.members,
            "is_centroid": self.cluster.is_centroid
        }  

        for feature in features:
            if feature in self.features:
                record.update(self.features[feature])
            
            elif isinstance(feature, Features):
                raise FeatureError(
                    f"Feature {feature.name!r} has not been calculated for the collection."
                )
            else:
                raise FeatureError(
                    f"Feature {feature!r} is not an instance of Features(Enum) class."
                )
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

    def __init__(self, molecule, prop_name="_Name") -> None:
        """[summary]

        Parameters
        ----------
        molecule : rdkit.Chem.Mol
            rdkit Mol instance
        prop_name : str, optional
            Contains the name of the molecule, which in turn contains
            the activity information, by default "_Name". 
            This is to maintain the current workflow (10/2021), but it's
            not practical.
        """
        super().__init__()
        self.molecule = molecule
        self.name = molecule.GetProp(prop_name)

        self.activity = "/active" in self.name
        self.is_external = (
            False  # Default is False, extracting external set will set to True
        )

class MoleculeFromSMILES(BaseMolecule):
    """Molecule initialized from SMILE. Must supply activity and name."""

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
        self.name = name

        self.activity = activity
        self.is_external = (
            False  # Default is False, extracting external set can set to True
        )
