from modtox.modtox.utils.enums import StandardTypes
from modtox.modtox.utils._custom_errors import BadRequestError, UnsupportedStandardRelation
from modtox.modtox.Molecules.act import Standard, Activity
from modtox.modtox.Retrievers.bindingdb import RetrieveBDB

import pytest
from unittest.mock import patch

def test_successful_request_by_target():
    r = RetrieveBDB()
    req = r._request_by_target("P23316")
    assert req.status_code == 200

def test_unsuccessful_request_by_target():
    r = RetrieveBDB()
    with pytest.raises(BadRequestError):
        req = r._request_by_target("randomstring")

def test_get_json():
    r = RetrieveBDB()
    req = r._request_by_target("P23316")
    r._get_unparsed_activities(req)
    assert r.unparsed_activities
    assert len(r.unparsed_activities) == 39

@pytest.mark.parametrize("std_type_i,std_val_i,std_type,std_rel,std_val,std_unit", [
    ("Ki", " 50", StandardTypes.Ki, "=", 50, "nM"),
    ("Ki", " <50", StandardTypes.Ki, "<", 50, "nM"),
    ("IC50", " >50", StandardTypes.IC50, ">", 50, "nM"),
])
def test_normalize_standard_successful(std_type_i,std_val_i,std_type,std_rel,std_val,std_unit):
    r = RetrieveBDB()
    a, b, c, d = r._normalize_standard(std_type_i, std_val_i)
    assert a == std_type
    assert b == std_rel
    assert c == std_val
    assert d == std_unit

def test_normalize_standard_error():
    r = RetrieveBDB()
    with pytest.raises(UnsupportedStandardRelation):
        a, b, c, d = r._normalize_standard(" randomstring", "50")

@patch("modtox.modtox.Retrievers.bindingdb.RetrieveBDB._normalize_standard")
def test_parse_activity(mock):
    mock.return_value = StandardTypes.IC50, "=", 70.0, "nM"
    unparsed_act = {
        'query': 'P23316', 'monomerid': 50089546, 'smile': 'CC(C)(C)C#C\\C=C\\CNc1c(=O)ccc12', 
        'affinity_type': 'IC50', 'affinity': 70, 'pmid': 10888332, 'doi': '10.1016/s0960-894x(00)00257-2'}
    
    r = RetrieveBDB()
    activity = r._parse_activity(unparsed_act)
    assert isinstance(activity, Activity)
    assert isinstance(activity.standard, Standard)
    assert r.ids[unparsed_act["smile"]] == f"BDBM{unparsed_act['monomerid']}"
    assert activity.target == unparsed_act["query"]

def test_retrieve_by_target_successful():
    r = RetrieveBDB()
    summ = r.retrieve_by_target("P23316")
    assert summ.request == "Successful"
    assert summ.retrieved_activities == 39
    assert summ.retrieved_molecules == 38

def test_retrieve_by_target_unsuccessful():
    r = RetrieveBDB()
    summ = r.retrieve_by_target("randomstring")
    assert summ.request == "No hits found"
    assert summ.retrieved_activities == 0
    assert summ.retrieved_molecules == 0