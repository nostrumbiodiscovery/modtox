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

class Features(Enum):
    glide = auto()  # glide features
    mordred = auto()  # mordred descriptors
    topo = auto()  # rdkit topological fingerprints
