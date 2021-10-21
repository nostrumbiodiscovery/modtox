from modtox.modtox.utils.enums import Features, StandardTypes
from modtox.modtox.Molecules.mol import MoleculeFromChem, MoleculeFromInChI
from modtox.modtox.utils._custom_errors import FeatureError
from modtox.modtox.Molecules.act import Activity, Standard
from modtox.modtox.utils._custom_errors import UnsupportedStandardType

import pytest

import os
from rdkit import Chem

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REDUCED_DATASET = os.path.join(DATA, "reduced_dataset")

ACTIVES = os.path.join(REDUCED_DATASET, "actives.sdf")
INACTIVES = os.path.join(REDUCED_DATASET, "inactives.sdf")
GLIDE_CSV = os.path.join(REDUCED_DATASET, "glide_features.csv")

OUTPUT_FOLDER = os.path.join(DATA, "tests_outputs")

actives = [
    mol for mol in Chem.SDMolSupplier(ACTIVES) if ":" in mol.GetProp("_Name")
]
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
    mol = MoleculeFromInChI(sm, True, "ibuprofen")
    mol.name
    assert isinstance(mol.molecule, Chem.Mol)


def test_add_cluster_info():
    mol = MoleculeFromChem(actives[0])
    mol.add_cluster_info(1, 2, True)
    assert mol.cluster.id == 1
    assert mol.cluster.members == 2
    assert mol.cluster.is_centroid == True


@pytest.mark.parametrize("std_type,std_val,expected",[
    (StandardTypes.Ki, 30, "Inactive"),
    (StandardTypes.Ki, 100, "Active"),
    (StandardTypes.Ki, 60, "Contradictory data"),
    (StandardTypes.IC50, 60, "No data"),
])
def test_assess_activity(std_type, std_val, expected):
    mol = MoleculeFromInChI("abc", "test_molecule")
    mol.add_activity(Activity("abc", Standard(StandardTypes.Ki, 50, "nM", "="), "placeholder", "placeholder"))
    mol.add_activity(Activity("abc", Standard(StandardTypes.Ki, 70, "nM", "="), "placeholder", "placeholder"))
    mol.assess_activity(std_type, std_val)
    assert mol.activity_tags[f"Ki, {std_val} nM"] == expected

def test_assess_activity_multiple():
    mol = MoleculeFromInChI("abc", "test_molecule")
    mol.add_activity(Activity("abc", Standard(StandardTypes.Ki, 100, "nM", "="), "placeholder", "placeholder"))
    mol.add_activity(Activity("abc", Standard(StandardTypes.IC50, 100, "nM", "="), "placeholder", "placeholder"))
    mol.assess_activity(StandardTypes.Ki, 50)
    mol.assess_activity(StandardTypes.IC50, 120)
    assert mol.activity_tags[f"Ki, 50 nM"] == "Inactive"
    assert mol.activity_tags[f"IC50, 120 nM"] == "Active"

def test_assess_activity_error():
    mol = MoleculeFromInChI("abc", "test_molecule")
    with pytest.raises(UnsupportedStandardType):
        mol.assess_activity("randomstring", 100)

def test_assess_activity_pubchem():
    mol = MoleculeFromInChI("abc", "test_molecule")
    mol.assess_activity("pubchem", 0)
    assert mol.ref_criterion == "pubchem"


def test_add_properties_to_molecule_after_activity_assessment():
    inchi = "InChI=1S/C3H6O/c1-3(2)4/h1-2H3"
    m = MoleculeFromInChI(inchi, "testmol")
    m.ref_criterion = "criterion_1"
    m.activity_tags = {"criterion_1": "Ki, 100 nM"}
    m._add_properties_to_molecule()
    assert m.molecule.GetProp("_Name") == "testmol"
    assert m.molecule.GetProp("InChI") == inchi
    assert m.molecule.GetProp("Activity") == "Ki, 100 nM"

def test_add_properties_to_molecule_before_activity_assessment():
    inchi = "InChI=1S/C3H6O/c1-3(2)4/h1-2H3"
    m = MoleculeFromInChI(inchi, "testmol")
    m.activity_tags = {}
    m._add_properties_to_molecule()
    assert m.molecule.GetProp("_Name") == "testmol"
    assert m.molecule.GetProp("InChI") == inchi
    with pytest.raises(KeyError):
        a = m.molecule.GetProp("Activity")