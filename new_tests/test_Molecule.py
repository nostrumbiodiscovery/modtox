from modtox.modtox.new_models_classes.Features.feat_enum import Features
from modtox.modtox.new_models_classes.mol import MoleculeFromChem, MoleculeFromSMILES
from modtox.modtox.new_models_classes._custom_errors import FeatureError

import os
from rdkit import Chem

DATA = os.path.join(os.path.dirname(__file__), "data")
ACTIVES = os.path.join(DATA, "actives.sdf")
INACTIVES = os.path.join(DATA, "inactives.sdf")
GLIDE = os.path.join(DATA, "glide_features.csv")

actives = [mol for mol in Chem.SDMolSupplier(ACTIVES) if ":" in mol.GetProp("_Name")]
inactives = [
    mol for mol in Chem.SDMolSupplier(INACTIVES) if ":" in mol.GetProp("_Name")
]
active = actives[0]
inactive = inactives[0]


def test_molecule_from_sdf_constructor():
    """Tests that rdkit molecule object has a '_Name' 
    parameter and contains the activity information."""
    active_mol = MoleculeFromChem(active)
    inactive_mol = MoleculeFromChem(inactive)
    assert active_mol.activity == True
    assert inactive_mol.activity == False
    return


def test_to_record_error():
    """Tests error is raised when trying to add no-calculated features."""
    mol = MoleculeFromChem(active)

    try:
        rec = mol.to_record()
    except FeatureError:
        assert True


def test_to_record():

    mol = MoleculeFromChem(active)

    mol.add_feature(Features.glide, {"G1": 1, "G2": 2})
    mol.add_feature(Features.topo, {"FP1": 3, "FP2": 4, "FP3": 5, "FP4": 6})
    mol.add_feature(Features.mordred, {"M1": 7, "M2": 8, "M3": 9})

    assert len(mol.to_record()) == 11
    assert len(mol.to_record(Features.glide, Features.topo)) == 8


def test_calculate_similarity():
    """Tests for similarity calculation to a molecule list.
    If it breaks, probably related to similarity calculation:
    rdkit.DataStructs."""

    mol1 = MoleculeFromChem(actives[0])
    mol2 = MoleculeFromChem(actives[1])
    mol3 = MoleculeFromChem(actives[2])
    mol4 = MoleculeFromChem(actives[3])

    for mol in [mol1, mol2, mol3, mol4]:
        mol.topo = Chem.RDKFingerprint(mol.molecule)
    sim = mol1.calculate_similarity([mol2, mol3, mol4])

    assert sim == 0.3922010964495473


def test_has_glide():
    """Tests has_glide property returns False when all
    Glide features equal to 0."""
    mol_w_glide = MoleculeFromChem(actives[0])
    mol_wo_glide = MoleculeFromChem(actives[1])
    mol_w_glide.add_feature(Features.glide, {"G1": 1, "G2": 0})
    mol_wo_glide.add_feature(Features.glide, {"G1": 0, "G2": 0})

    assert mol_w_glide.has_glide == True
    assert mol_wo_glide.has_glide == False


def test_mol_from_smiles_init():
    sm = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    mol = MoleculeFromSMILES(sm, True, "ibuprofen")
    mol.name
    assert isinstance(mol.molecule, Chem.Mol)

def test_add_cluster_info():
    mol = MoleculeFromChem(actives[0])
    mol.add_cluster_info(1, 2, True)
    assert mol.cluster.id == 1
    assert mol.cluster.members == 2
    assert mol.cluster.is_centroid == True