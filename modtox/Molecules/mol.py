from modtox.modtox.Molecules.act import Activity
from modtox.modtox.utils.enums import Features, StandardTypes
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Collection, Dict, List
from dataclasses import dataclass
from modtox.modtox.utils._custom_errors import FeatureError
import modtox.modtox.constants.constants as k
from modtox.modtox.utils._custom_errors import UnsupportedStandardType

@dataclass
class Cluster:
    """Dataclass for storing clustering information. 
    """
    id: int
    members: int
    is_centroid: bool


class BaseMolecule:
    """Represents the base molecule. Should not instantiated."""

    molecule: Chem.Mol
    features: Dict[Features, Dict[str, float or int]]
    scaffold: str

    def __init__(self) -> None:
        """Creates empty feature dictionary to be populated as
        Features.xxx: {"F1": 1, "F2": 0.5, ...}, Features.yyy: {...}.
        Creates empty similarity list. 
        """
        self.features = dict()

    def add_cluster_info(
        self, cluster_id: int, members: int, is_centroid: bool
    ):
        """Adds the cluster information calculated in the collection.

        Parameters
        ----------
        cluster_id : int
            Cluster index (arbitrary value). Not used for the moment.
        members : int
            Number of members in the cluster of the molecule. Used for removing
            outliers.
        is_centroid : bool
            If the molecule is the centroid of its cluster.
        """
        self.cluster = Cluster(cluster_id, members, is_centroid)

    def add_feature(self, ft: Features, d: Dict[str, float or int]):
        """Adds a dictionary of features to the molecule's features dictionary.

        Parameters
        ----------
        ft : Features (enum)
            Instance of the Enum Features class being added.
        d : Dict[str, float or int]
            Formatted as as: {'F1': 0, 'F2': 1.1, ...}"""

        self.features[ft] = d

    def to_record(self, *features: Features):
        """Converts the molecule to a record format. Contains the activity,
        cluster info and the specified features.If no *args are supplied, 
        includes all calculated features.

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
            When the specified Feature is not in the
            self.features or is not even a Features(Enum) instance. 
        """

        if not features:
            features = list(self.features.keys())

        record = {
            "Activity": self.activity,
            "cluster_id": self.cluster.id,
            "cluster_members": self.cluster.members,
            "is_centroid": self.cluster.is_centroid,
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
    '_Name' property containing activity information. Currently used to
    create molecules after reading SDF (SDMolSupplier)"""

    name: str
    activity: bool

    def __init__(self, molecule, prop_name="_Name") -> None:
        """[summary]

        Parameters
        ----------
        molecule : rdkit.Chem.Mol
            rdkit Mol instance
        prop_name : str, optional
            Contains the name of the molecule, which in turn contains
            the activity information, by default "_Name". 
        """
        super().__init__()
        self.molecule = molecule
        self.name = molecule.GetProp(prop_name)

        self.activity = "/active" in self.name
        self.is_external = (
            False  # Default is False, extracting external set will set to True
        )


class MoleculeFromInChI(BaseMolecule):
    """Molecule initialized from InChI. Must supply InChI and name.
    Currently used to create molecules after retrieving from database."""

    molecule: Chem
    name: str or None

    activity: bool
    activities: List[Activity]
    features: Dict[Features, Dict[str, float or int]]

    def __init__(self, inchi: str, name: str = None) -> None:
        """Initializes instance and defines activities (list of 
        modtox.Activities) as well as activity tags {criterion -> activity}
        e.g.: {"Ki, 100 nm": "Active"}

        Parameters
        ----------
        inchi : str
            InChI.
        name : str, optional
            Currently not used, by default None
        """
        super().__init__()
        self.inchi = inchi
        self.molecule = Chem.MolFromInchi(inchi)
        self.name = name
        self.activities = list()
        self.activity_tags = dict()

    def add_activity(self, activity):
        """Adds modtox.Activity object to the list of activities.

        Parameters
        ----------
        activity : Activity
            Activiy object.
        """
        self.activities.append(activity)

    def assess_activity(self, standard_type: StandardTypes, threshold: float):
        """Catalogues the molecule in "Active", "Inactive", "No data" 
        or "Contradictory" It sets a ref_criterion (first criterion provided).
        PubChem tagged activity is added in the collection.
    
        Parameters
        ----------
        standard_type : StandardTypes
            Type of standard assessed. 
        threshold : float
            Threshold of active/inactive.

        Raises
        ------
        UnsupportedStandardType
            If standard type has not been defined in enum class.
        """
        if standard_type == "pubchem":
            if not self.activity_tags:  # Means it's the first criterion supplied
                self.ref_criterion = "pubchem"  
            return  # Returns None, added by modtox.CollectionFromTarget
        
        if standard_type not in StandardTypes:
            raise UnsupportedStandardType(f"Standard {standard_type!r} not supported.")
        
        units = k.bdb_units[standard_type.name]

        if not self.activity_tags:  # Means it's the first criterion supplied
            self.ref_criterion = f"{standard_type.name}, {threshold} {units}"

        is_active = list()
        
        for activity in self.activities:
            if activity.standard.std_type == standard_type and activity.standard.std_val <= threshold:
                is_active.append(True)
            elif activity.standard.std_type == standard_type and activity.standard.std_val > threshold:
                is_active.append(False)
        
        if not is_active:
            self.activity_tags[f"{standard_type.name}, {threshold} {units}"] = "No data"
        elif all(is_active):
            self.activity_tags[f"{standard_type.name}, {threshold} {units}"] = "Active"
        elif any(is_active) == False:
            self.activity_tags[f"{standard_type.name}, {threshold} {units}"] = "Inactive"
        else:
            self.activity_tags[f"{standard_type.name}, {threshold} {units}"] = "Contradictory data"

    def _add_properties_to_molecule(self):
        """Adds the properties to the rdkit.Mol object to keep the information 
        between exporting to SDF.
        """
        Chem.AssignStereochemistry(self.molecule)
        AllChem.EmbedMolecule(self.molecule)
        self.molecule.SetProp("_MolFileChiralFlag","1")
        self.molecule.SetProp("_Name", self.name)
        self.molecule.SetProp("InChI", self.inchi)
        try:
            self.molecule.SetProp("Activity", self.activity_tags.get(self.ref_criterion))
        except AttributeError:
            pass  # It's ugly, but maybe someone wants to export a collection without
                  # tagging activity.
