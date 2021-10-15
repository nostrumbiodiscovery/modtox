from enum import Enum, auto


class Features(Enum):
    glide = auto()  # glide features
    mordred = auto()  # mordred descriptors
    topo = auto()  # rdkit topological fingerprints
