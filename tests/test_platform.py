import pytest
import os
import glob
import modtox.main as mn

DATA_PATH=os.path.join(os.path.dirname(__file__), "data")
TRAJ=os.path.join(DATA_PATH, "general/traj.pdb")
TOP=os.path.join(DATA_PATH, "general/init.top")
ACTIVE=os.path.join(DATA_PATH, "active_decoys/active.sdf")
INACTIVE=os.path.join(DATA_PATH, "active_decoys/decoys.sdf")
DUDE=os.path.join(DATA_PATH, "dude")
PUBCHEM = os.path.join(DATA_PATH, "pubchem")
CSV_FILENAME = "AID_1851_datatable_all.csv"
SUBSTRATE = "p450-cyp2c9"
STATUS = "train"
FILENAME_MODEL = os.path.join(DATA_PATH, "analysis/fitted_models.pkl")
GLIDE_FILES=os.path.join(DATA_PATH, "analysis/input__*dock_lib.maegz")
RESNAME="198"
ACTIVE_ANALYSIS=os.path.join(DATA_PATH, "analysis/active.sdf")
INACTIVE_ANALYSIS=os.path.join(DATA_PATH, "analysis/decoys.sdf")

@pytest.mark.parametrize("traj, resname, top, active, inactive", [
                         (TRAJ, RESNAME, TOP, ACTIVE, INACTIVE),
                         ])
def test_docking(traj, resname, top, active, inactive):
     mn.main([traj,], resname, active, inactive, top=top, dock=True, sieve=1, debug=True)


@pytest.mark.parametrize("traj, resname, top, dude, status", [
                         (TRAJ, RESNAME, TOP, DUDE, STATUS),
                         ])
def test_dude(traj, resname, top, dude, status):
     mn.main([traj,], resname, dude=dude, status = status, top=top, dock=True, debug=True)

@pytest.mark.parametrize("traj, resname, top, pubchem, csv_filename, substrate, status", [
                         (TRAJ, RESNAME, TOP, PUBCHEM, CSV_FILENAME, SUBSTRATE, STATUS),
                         ])
def test_pubchem(traj, resname, top, pubchem, csv_filename, substrate, status):
     mn.main([traj,], resname, pubchem=pubchem, csv_filename = csv_filename, substrate = substrate,top=top, status = status, dock=True, debug=True)

@pytest.mark.parametrize("traj, resname, top, active, inactive, filename_model", [
                         (TRAJ, RESNAME, TOP, ACTIVE_ANALYSIS, INACTIVE_ANALYSIS, FILENAME_MODEL),
                         ])
def test_model_stack(traj, resname, top, active, inactive, filename_model):
     initial_dir = os.getcwd()
     os.chdir(os.path.join(DATA_PATH, "analysis"))
     mn.main(traj, resname, active, inactive, top=top, assemble_model=True, filename_model = filename_model, glide_files=GLIDE_FILES, debug=True, cv=2, classifier="stack")
     os.chdir(initial_dir)

@pytest.mark.parametrize("traj, resname, top, active, inactive, filename_model", [
                         (TRAJ, RESNAME, TOP, ACTIVE_ANALYSIS, INACTIVE_ANALYSIS, FILENAME_MODEL),
                         ])
def test_model_normal(traj, resname, top, active, inactive, filename_model):
     initial_dir = os.getcwd()
     os.chdir(os.path.join(DATA_PATH, "analysis"))
     mn.main(traj, resname, active, inactive, top=top, assemble_model=True, filename_model = filename_model, glide_files=GLIDE_FILES, debug=True, cv=2)
     os.chdir(initial_dir)

@pytest.mark.parametrize("traj, resname, top, active, inactive, dude, filename_model", [
                         (TRAJ, RESNAME, TOP, ACTIVE_ANALYSIS, INACTIVE_ANALYSIS, DUDE, FILENAME_MODEL),
                         ])
def test_predict_dude_normal(traj, resname, top, active, inactive, dude, filename_model):
     initial_dir = os.getcwd()
     os.chdir(os.path.join(DATA_PATH, "analysis"))
     mn.main(traj, resname, active, inactive, top=top,filename_model = filename_model, predict = True)
     os.chdir(initial_dir)

@pytest.mark.parametrize("traj, resname, top, active, inactive, dude, filename_model", [
                         (TRAJ, RESNAME, TOP, ACTIVE_ANALYSIS, INACTIVE_ANALYSIS, DUDE, FILENAME_MODEL),
                         ])
def test_predict_dude_stack(traj, resname, top, active, inactive, dude, filename_model):
     initial_dir = os.getcwd()
     os.chdir(os.path.join(DATA_PATH, "analysis"))
     mn.main(traj, resname, active, inactive, top=top,filename_model = filename_model, classifier = "stack", predict = True)
     os.chdir(initial_dir)
