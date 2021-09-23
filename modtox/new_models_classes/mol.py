from mordred import Calculator, descriptors
from rdkit import Chem
from typing import Dict


class ModelError(Exception):
    pass

class Molecule:
    """Object representing a molecule."""

    name: str
    activity: bool
    molecule: Chem
    glide: Dict[str, float]
    topo: Dict[str, float]
    mordred: Dict[str, float]
    
    def __init__(self, molecule) -> None:
        self.molecule = molecule
        self.name = molecule.GetProp("_Name")
        self.activity = "/active" in self.name
    
    def calculate_topo(self):
        topo = Chem.RDKFingerprint(self.molecule)
        d = { f"rdkit_fp_{i}": int(fp) for i, fp in enumerate(topo) }
        self.topo = d
        return

    def calculate_mordred(self):
        mord = Calculator(descriptors, ignore_3D=True).pandas([self.molecule], quiet=True)
        self.mordred = mord.to_dict("index")[0]
        return

    def to_record(self, glide=True, topo=True, mordred=True):
        record = {
            "Activity": self.activity,
        }
        if glide:
            record.update(self.glide)
        if topo:
            record.update(self.topo)
        if mordred:
            record.update(self.mordred)
        return record



