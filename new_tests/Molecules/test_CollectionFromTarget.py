from modtox.modtox.Molecules.target_col import CollectionFromTarget, IDs
from modtox.modtox.Molecules.mol import MoleculeFromInChI
from modtox.modtox.utils.enums import Database

from unittest.mock import MagicMock, Mock, PropertyMock, patch


@patch(
    "modtox.modtox.Molecules.col.RetrieveBDB.activities",
    create=True,
    new_callable=PropertyMock,
)
@patch(
    "modtox.modtox.Molecules.col.RetrieveBDB.ids",
    create=True,
    new_callable=PropertyMock,
)
@patch("modtox.modtox.Molecules.col.RetrieveBDB.retrieve_by_target")
@patch(
    "modtox.modtox.Molecules.col.RetrieveChEMBL.activities",
    create=True,
    new_callable=PropertyMock,
)
@patch(
    "modtox.modtox.Molecules.col.RetrieveChEMBL.ids",
    create=True,
    new_callable=PropertyMock,
)
@patch("modtox.modtox.Molecules.col.RetrieveChEMBL.retrieve_by_target")
@patch(
    "modtox.modtox.Molecules.col.RetrievePubChem.activities",
    create=True,
    new_callable=PropertyMock,
)
@patch(
    "modtox.modtox.Molecules.col.RetrievePubChem.ids",
    create=True,
    new_callable=PropertyMock,
)
@patch("modtox.modtox.Molecules.col.RetrievePubChem.retrieve_by_target")
def test_fetch(
    pc_ret,
    pc_ids,
    pc_acts,
    ch_ret,
    ch_ids,
    ch_acts,
    bdb_ret,
    bdb_ids,
    bdb_acts,
):
    pc_ids.return_value = {"sm1": "cid1", "sm2": "cid2"}
    pc_acts.return_value = ["pc_act1", "pc_act2", "pc_act3"]
    ch_ids.return_value = {"sm1": "ch1", "sm3": "ch2"}
    ch_acts.return_value = ["ch_act1", "ch_act2", "ch_act3"]
    bdb_ids.return_value = {"sm2": "bdbm1", "sm4": "bdbm2"}
    bdb_acts.return_value = ["bdb_act1", "bdb_act2", "bdb_act3"]

    c = CollectionFromTarget("abc")
    c.fetch([e for e in Database])
    
    assert c.activities == [
        "bdb_act1", "bdb_act2", "bdb_act3", 
        "pc_act1", "pc_act2", "pc_act3", 
        "ch_act1", "ch_act2", "ch_act3"
    ]

    assert c.ids == {
        Database.BindingDB: {"sm2": "bdbm1", "sm4": "bdbm2"},
        Database.PubChem: {"sm1": "cid1", "sm2": "cid2"},  
        Database.ChEMBL: {"sm1": "ch1", "sm3": "ch2"}, 
    }

def test_unique_smiles():
    c = CollectionFromTarget("abc")
    c.ids = {
        "db1": {"sm1": "id1", "sm2": "id2"},
        "db2": {"sm1": "id3", "sm3": "id4"},
        "db3": {"sm4": "id5", "sm1": "id6"},
    }
    c.get_unique_inchis()
    assert c.unique_inchis == {"sm1", "sm2", "sm3", "sm4"}


def test_unify_ids():
    c = CollectionFromTarget("abc")
    c.ids = {
        Database.BindingDB: {"sm1": "id1", "sm2": "id2"},
        Database.ChEMBL: {"sm1": "id3", "sm3": "id4"},
    }
    c.unique_inchis = {"sm1", "sm2", "sm3"}
    
    c._unify_ids()

    for obj in c.unified_ids.values():
        assert isinstance(obj, IDs)
    assert c.unified_ids["sm1"].bindingdb_id == "id1"
    assert c.unified_ids["sm1"].chembl_id == "id3"
    assert c.unified_ids["sm2"].bindingdb_id == "id2"
    assert c.unified_ids["sm3"].chembl_id == "id4"
    

def test_create_molecules():
    class MockAct:
        def __init__(self, smiles) -> None:
            self.smiles = smiles

    c = CollectionFromTarget("abc")
    c.unique_inchis = ["sm1", "sm2", "sm3"]  # As a list instead of set to maintain order
    c.activities = [
        MockAct("sm1"),
        MockAct("sm2"),
        MockAct("sm3"),
        MockAct("sm1"),
        MockAct("sm1"),
        MockAct("sm2"),
    ]

    c.unified_ids = {"sm1": IDs(), "sm2": IDs(), "sm3": IDs()}
    c.create_molecules()
    assert len(c.molecules) == len(c.unique_inchis)
    assert isinstance(c.molecules[0].ids, IDs)
    assert len(c.molecules[0].activities) == 3
    assert len(c.molecules[1].activities) == 2
    assert len(c.molecules[2].activities) == 1

    assert c.molecules[0].name == "abc_0"
    assert c.molecules[1].name == "abc_1"
    assert c.molecules[2].name == "abc_2"


def test_integration_test():
    c = CollectionFromTarget("P23316")
    c.fetch([Database.BindingDB, Database.ChEMBL, Database.PubChem])
    c.get_unique_inchis()
    c._unify_ids()
    c.create_molecules()
    return

@patch("modtox.modtox.Molecules.mol.MoleculeFromInChI.assess_activity")
def test_asses_activity(mock):
    c = CollectionFromTarget("P23316")
    c.molecules = [MoleculeFromInChI("hi"), MoleculeFromInChI("hi")]
    criteria = {"Ki": 10, "IC50": 30}
    c.assess_activity(criteria)
    assert mock.call_count == 4

def test_add_pubchem_activity():
    c = CollectionFromTarget("P23316")
    c.molecules = [MoleculeFromInChI("inchi1", "mol1"), MoleculeFromInChI("inchi2", "mol2"), MoleculeFromInChI("inchi3, mol3")]
    c.pubchem_activity = {"inchi1": "act1", "inchi2": "act2"}
    c.add_pubchem_activity()
    assert c.molecules[0].activity_tags["pubchem"] == "act1"
    assert c.molecules[1].activity_tags["pubchem"] == "act2"
    assert c.molecules[2].activity_tags["pubchem"] is None

@patch("modtox.modtox.Molecules.target_col.CollectionFromTarget.add_pubchem_activity")
def test_assess_acticity_pubchem(mock):
    c = CollectionFromTarget("P23316")
    c.assess_activity({"pubchem": 0})
    mock.assert_called_once_with()