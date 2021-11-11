from modtox.modtox.Molecules.target_col import CollectionFromTarget
from modtox.modtox.utils.enums import Database

def test_retrieve():
    """For debugging purposes"""
    c = CollectionFromTarget("P00740")
    dbs = [Database.BindingDB, Database.ChEMBL, Database.PubChem]
    c.fetch(dbs)