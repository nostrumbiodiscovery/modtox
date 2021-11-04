from modtox.modtox.utils.enums import StandardTypes
from modtox.modtox.utils._custom_errors import UnsupportedStandardType
from modtox.modtox.Molecules.act import Standard, Activity
from modtox.modtox.Retrievers.pubchem import RetrievePubChem
from modtox.modtox.utils import utils as u

import os
import pubchempy
import json

import pytest
from unittest.mock import patch

def test_download_csv_successful():
    r = RetrievePubChem()
    confirmation = r._download_json("P23316")
    assert confirmation == "Successful"
    assert u.get_latest_file("*") == os.path.join(os.getcwd(), "PROTACXN_P23316_bioactivity_protein.json")

@pytest.mark.parametrize("input,expected", 
    [(19, "C1=CC(=C(C(=C1)O)O)C(=O)O"),
    (33, "C(C=O)Cl"),
])
def test_cid2smiles(input, expected):
    r = RetrievePubChem()
    assert r._cid2smiles(input) == expected

def test_cid2_smiles_unsuccessful():
    r = RetrievePubChem()
    with pytest.raises(pubchempy.BadRequestError):
        smiles = r._cid2smiles(891744216472167260)

def test_read_json(): 
    d = {"this": 0, "is": 1, "a": 2, "test": 3}
    with open("test.json", "w") as f:
        json.dump(d, f)
    afile = os.path.join(os.getcwd(), "test.json")

    r = RetrievePubChem()
    js = r._read_json()
    assert u.get_latest_file("*json") != afile
    assert js == d


def test_normalize_standard_successful():
    unparsed_activity = {
		"acname": "Ki",
		"acqualifier": "=",
		"acvalue": "11.3",
    }
    r = RetrievePubChem()
    std = r._normalize_standard(unparsed_activity)
    assert std.std_type == StandardTypes.Ki
    assert std.std_rel == unparsed_activity["acqualifier"]
    assert std.std_val == float(unparsed_activity["acvalue"]) * 1000
    assert std.std_unit == "nM"

def test_normalize_standard_unsucessful():
    unparsed_activity = {
		"acname": "randomstring",
		"acqualifier": "=",
		"acvalue": "11.3",
    }
    r = RetrievePubChem()
    with pytest.raises(UnsupportedStandardType):    
        std = r._normalize_standard(unparsed_activity)

@patch("modtox.modtox.Retrievers.pubchem.RetrievePubChem._normalize_standard")
def test_parse_activity(mock):
    mock.return_value = Standard(StandardTypes.Ki, "=", 11300.0, "nM")
    r = RetrievePubChem()
    unparsed_act = {
		"baid": "99548774",
		"activityid": "Active",
		"aid": "52104",
		"sid": "134437925",
		"cid": "56661147",
		"pmid": "2939243",
		"aidtypeid": "Confirmatory",
		"aidmdate": "20181011",
		"hasdrc": "0",
		"rnai": "0",
		"protacxn": "P23316",
		"acname": "Ki",
		"acqualifier": "=",
		"acvalue": "11.3",
		"aidsrcname": "ChEMBL",
		"aidname": "Compound was tested for the inhibitory activity against chitin synthetase",
		"cmpdname": "2-[[(E)-2-[(2-Amino-3-phenylpropanoyl)amino]-3-phenylprop-2-enoyl]amino]-2-[(2R,3S,4R,5R)-5-(2,4-dioxopyrimidin-1-yl)-3,4-dihydroxyoxolan-2-yl]acetic acid",
		"targetname": "Chitin synthase 1 (Candida albicans)",
		"targeturl": "/protein/P23316",
		"ecs": ["2.4.1.16"],
		"repacxn": "P23316",
		"taxids": [5476],
		"targettaxid": ""
    }
    activity = r._parse_activity(unparsed_act)
    assert isinstance(activity, Activity)
    assert isinstance(activity.standard, Standard)
    assert r.ids[activity.smiles] == "56661147"
    assert activity.target == "P23316"

def test_retrieve_by_target_successful():
    r = RetrievePubChem()
    summ = r.retrieve_by_target("P23316")
    assert summ.request == "Successful"
    assert summ.retrieved_activities == 80
    assert summ.retrieved_molecules == 43

def test_retrieve_by_target_unsuccessful():
    r = RetrievePubChem()
    summ = r.retrieve_by_target("randomstring")
    assert summ.request == "No hits found"
    assert summ.retrieved_activities == 0
    assert summ.retrieved_molecules == 0