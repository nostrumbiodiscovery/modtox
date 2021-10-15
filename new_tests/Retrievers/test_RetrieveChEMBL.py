from modtox.modtox.utils.enums import StandardTypes
from modtox.modtox.utils._custom_errors import BadRequestError, UnsupportedStandardType
from modtox.modtox.Molecules.act import Standard, Activity
from modtox.modtox.Retrievers.chembl import RetrieveChEMBL

import pytest
from unittest.mock import patch

@pytest.mark.parametrize(
    "input,expected",[
    ("P23316", "CHEMBL3480"),
    ("P18846", "CHEMBL3255"),
    ("P08246", "CHEMBL248"),
])
def test_uniprot2chembl_successful(input, expected):
    r = RetrieveChEMBL()
    assert r._uniprot2chembl(input) == expected

def test_uniprot2chembl_unsucessful():
    r = RetrieveChEMBL()
    with pytest.raises(BadRequestError):
        r._uniprot2chembl("P10862")  # Target with no hits


def test_request_by_target():
    r = RetrieveChEMBL()
    unparsed_activities = r._request_by_target("CHEMBL3480")
    assert len(unparsed_activities) == 106
    assert len(unparsed_activities[0]) == 6

def test_normalize_standard_sucessful():
    r = RetrieveChEMBL()
    unparsed_activity = {
        'canonical_smiles': 'NC(Cc1ccc(O)cc1)C(=O...O)[C@@H]1O', 
        'molecule_chembl_id': 'CHEMBL1791412', 'standard_type': 'ID50', 'standard_relation': '=', 
        'standard_units': 'M', 'standard_value': '0.0000049'}
    std = r._normalize_standard(unparsed_activity)
    assert std.std_type == StandardTypes.ID50
    assert std.std_rel == unparsed_activity["standard_relation"]
    assert std.std_val == float(unparsed_activity["standard_value"])
    assert std.std_unit == unparsed_activity["standard_units"]

def test_normalize_standard_error():
    r = RetrieveChEMBL()
    unparsed_activity = {
        'canonical_smiles': 'NC(Cc1ccc(O)cc1)C(=O...O)[C@@H]1O', 
        'molecule_chembl_id': 'CHEMBL1791412', 'standard_type': 'randomstring', 'standard_relation': '=', 
        'standard_units': 'M', 'standard_value': '0.0000049'}
    with pytest.raises(UnsupportedStandardType):
        r._normalize_standard(unparsed_activity)

@patch("modtox.modtox.Retrievers.chembl.RetrieveChEMBL._normalize_standard", autospec=True)
def test_parse_activity(mock):
    mock.return_value = Standard(StandardTypes.ID50, "=", 0.0000049, "M")
    r = RetrieveChEMBL()
    r.target = "abc"
    unparsed_act = {
        'canonical_smiles': 'NC(Cc1ccc(O)cc1)C(=O...O)[C@@H]1O', 
        'molecule_chembl_id': 'CHEMBL1791412', 'standard_type': 'randomstring', 'standard_relation': '=', 
        'standard_units': 'M', 'standard_value': '0.0000049'}
    activity = r._parse_activity(unparsed_act)
    assert isinstance(activity, Activity)
    assert isinstance(activity.standard, Standard)
    assert r.ids[unparsed_act["canonical_smiles"]] == "CHEMBL1791412"
    assert activity.target == "abc"

def test_retrieve_by_target_successful():
    r = RetrieveChEMBL()
    summ = r.retrieve_by_target("P23316")
    assert summ.request == "Successful"
    assert summ.retrieved_activities == 106
    assert summ.retrieved_molecules == 66

def test_retrieve_by_target_unsuccessful():
    r = RetrieveChEMBL()
    summ = r.retrieve_by_target("randomstring")
    assert summ.request == "No hits found"
    assert summ.retrieved_activities == 0
    assert summ.retrieved_molecules == 0