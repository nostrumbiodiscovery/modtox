import glob
import os
import pandas as pd
import pytest
import time

from modtox.docking.glide import glide, analyse
from .config import check_remove_folder

data_dir = "data"
sdf_active_train = os.path.join(data_dir, "actives.sdf")
sdf_inactive_train = os.path.join(data_dir, "inactives.sdf")


@pytest.mark.parametrize("greasy", [True, False])
def test_glide_docking(greasy):
    """
    Tests Glide docking - both greasy and not.
    """
    systems = glob.glob(os.path.join(data_dir, "docking", "cluster*.pdb"))
    docking_obj = glide.Glide_Docker(
        systems=systems,
        ligands_to_dock=[sdf_active_train, sdf_inactive_train],
        greasy=greasy,
        debug=False,
    )
    docking_obj.dock(precision="SP", maxkeep=500, maxref=400, grid_mol=2)
    time.sleep(25)  # it takes a while for the files to show up in the working directory
    outputs = glob.glob("./input*cluster*lig.maegz")
    assert len(outputs) == len(systems)

    # TODO: Add proper file clean-up, not only the ones we test for.


def test_glide_analysis():
    """
    Tests Glide analysis and checks for CSV with extracted features.
    """
    analysis_folder = "glide_analysis"
    systems = glob.glob(os.path.join(data_dir, "docking", "*dock_lib.maegz"))
    csv = analyse.analyze(
        systems,
        analysis_folder,
        best=False,
        csv=False,
        active=sdf_active_train,
        inactive=sdf_inactive_train,
        debug=False,
    )

    df = pd.read_csv(csv)
    assert len(df) == 195

    check_remove_folder(analysis_folder)
