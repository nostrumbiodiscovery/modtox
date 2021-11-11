from modtox.modtox.Molecules.col import CollectionFromTarget
from modtox.modtox.utils.enums import Database

from unittest.mock import PropertyMock, patch

@patch("modtox.modtox.Molecules.col.RetrieveBDB.activities", create=True, new_callable=PropertyMock)
@patch("modtox.modtox.Molecules.col.RetrieveBDB.ids", create=True, new_callable=PropertyMock)
@patch("modtox.modtox.Molecules.col.RetrieveBDB.retrieve_by_target")

@patch("modtox.modtox.Molecules.col.RetrieveChEMBL.activities", create=True, new_callable=PropertyMock)
@patch("modtox.modtox.Molecules.col.RetrieveChEMBL.ids", create=True, new_callable=PropertyMock)
@patch("modtox.modtox.Molecules.col.RetrieveChEMBL.retrieve_by_target")

@patch("modtox.modtox.Molecules.col.RetrievePubChem.activities", create=True, new_callable=PropertyMock)
@patch("modtox.modtox.Molecules.col.RetrievePubChem.ids", create=True, new_callable=PropertyMock)
@patch("modtox.modtox.Molecules.col.RetrievePubChem.retrieve_by_target")

def test_fetch(pc_ret, pc_ids, pc_acts, ch_ret, ch_ids, ch_acts, bdb_ret, bdb_ids, bdb_acts):
    pc_ids.return_value = {"sm1": "cid1", "sm2": "cid2"}
    pc_acts.return_value = ["pc_act1", "pc_act2", "pc_act3"] 
    ch_ids.return_value =  {"sm1": "ch1", "sm3": "ch2"}
    ch_acts.return_value = ["ch_act1", "ch_act2", "ch_act3"]
    bdb_ids.return_value = {"sm2": "bdbm1", "sm4": "bdbm2"}
    bdb_acts.return_value = ["bdb_act1", "bdb_act2", "bdb_act3"]

    c = CollectionFromTarget("abc")
    c.fetch([e for e in Database])

def test_unique_smiles():
    c = CollectionFromTarget("abc")
    c.ids = {
        "db1": {"sm1": "id1", "sm2": "id2"},
        "db2": {"sm1": "id3", "sm3": "id4"},
        "db3": {"sm4": "id5", "sm1": "id6"},
    }
    assert c.unique_smiles == {"sm1", "sm2", "sm3", "sm4"}