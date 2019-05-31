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
GLIDE_FILES=os.path.join(DATA_PATH, "analysis/input__*dock_lib.maegz")
RESNAME="198"

@pytest.mark.parametrize("traj, resname, top, active, inactive", [
                         (TRAJ, RESNAME, TOP, ACTIVE, INACTIVE),
                         ])
def test_docking(traj, resname, top, active, inactive):
     mn.main([traj,], resname, active, inactive, top=top, dock=True, sieve=1, debug=True)


@pytest.mark.parametrize("traj, resname, top, dude", [
                         (TRAJ, RESNAME, TOP, DUDE),
                         ])
def test_dude(traj, resname, top, dude):
     mn.main([traj,], resname, dude=dude, top=top, dock=True, debug=True)

@pytest.mark.parametrize("traj, resname, top, active, inactive", [
                         (TRAJ, RESNAME, TOP, ACTIVE, INACTIVE),
                         ])
def test_model(traj, resname, top, active, inactive):
     mn.main(traj, resname, active, inactive, top=top, analysis=True, glide_files=GLIDE_FILES)
