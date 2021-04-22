import filecmp
import os

from modtox.data.chembl import ChEMBL
from .config import retrieve_database_bindingdb, retrieve_database_dude, retrieve_database_pubchem, check_remove_folder


data_dir = "data"


def test_dude():
    """
    Tests DUDe database preprocessing.
    """
    output_folder = "dude_db"
    data = os.path.join(data_dir, "cp2c9")

    database = retrieve_database_dude(data, tmp=output_folder)
    sdf_active_train, sdf_inactive_train = database.process_dude()

    assert sdf_active_train and sdf_inactive_train
    assert os.path.exists(output_folder)
    assert filecmp.cmp(os.path.join(output_folder, 'used_mols.txt'), os.path.join(data_dir, 'used_mols_dude.txt'))

    check_remove_folder(output_folder)


def test_pubchem():
    """
    Tests PubChem database preprocessing.
    """
    n_molecules = 10
    output_folder = "pubchem_db"
    data = os.path.join(data_dir, "AID_1851_datatable_all.csv")

    database = retrieve_database_pubchem(data, "p450-cyp2c9", n_molecules, tmp=output_folder)
    sdf_active_test, sdf_inactive_test = database.process_pubchem()

    assert sdf_active_test and sdf_inactive_test
    assert os.path.exists(output_folder)
    assert filecmp.cmp(os.path.join(output_folder, 'used_mols.txt'), os.path.join(data_dir, 'used_mols_pubchem.txt'))

    check_remove_folder(output_folder)


def test_bindingdb():
    """
    Tests Binding Database preprocessing.
    """
    output_folder = "binding_db"
    data = os.path.join(data_dir, "cyp2c9_bindingdb.sdf")

    database = retrieve_database_bindingdb(binding=data, tmp=output_folder)
    sdf_active_train, sdf_inactive_train = database.process_bind()

    assert sdf_active_train and sdf_inactive_train
    assert os.path.exists(output_folder)
    assert filecmp.cmp(os.path.join(output_folder, 'used_mols.txt'), os.path.join(data_dir, 'used_mols_bindingdb.txt'))

    check_remove_folder(output_folder)


def test_chembl():
    """
    Tests extraction of actives and inactives SDFs from a ChEMBL CSV.
    """
    data = os.path.join(data_dir, "P07711.csv")
    output = "chembl_test"
    threshold = 100
    parser = ChEMBL(csv=data, folder_output=output, threshold=threshold)
    actives, inactives = parser.get_data()

    output_sdfs = [os.path.join(output, "actives_sanitized.sdf"), os.path.join(output, "inactives_sanitized.sdf")]

    for sdf in output_sdfs:
        assert os.path.exists(sdf)
