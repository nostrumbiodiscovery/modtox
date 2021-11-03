from enum import Enum, auto


class Database(Enum):
    """Enum class for databases"""

    BindingDB = auto()
    PubChem = auto()
    ChEMBL = auto()


class StandardTypes(Enum):
    """Enum class for standard types. UnsupportedStandardType will be 
    raised if trying to add an activity with a standard not listed here."""

    IC50 = auto()
    Ki = auto()
    Kd = auto()
    Inhibition = auto()
    ID50 = auto()
    EC50 = auto()
    Kobs_1 = auto()
    Efficacy = auto()
    Papp = auto()
    Activity = auto()



class Features(Enum):
    glide = auto()  # glide features
    mordred = auto()  # mordred descriptors
    topo = auto()  # rdkit topological fingerprints
    morgan = auto()  # rdkit morgan fingerprints

