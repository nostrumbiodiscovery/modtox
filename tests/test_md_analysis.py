import glob
import os
import pytest

from modtox.cpptraj import analysis
from .config import check_file, check_remove_folder

md_directory = "data/md_trajectories"


@pytest.mark.parametrize(
    ("data", "residue", "expected_line"),
    [
        (
            os.path.join(md_directory, "2qyk_holo"),
            "NPV",
            "#Clustering: 10 clusters 6566 frames",
        ),
        (
            os.path.join(md_directory, "3nxu_holo"),
            "RIT",
            "",
        ),
    ],
)
def test_analyse(data, residue, expected_line):
    """
    Tests analyse function in the cpptraj module, asserting the presence of clusters, info.dat and plot.
    """
    topology = os.path.join(data, "*.top")
    trajectory = os.path.join(data, "*.x")
    output_dir = "analysis_{}".format(residue)

    # TODO: What happens with APO?

    # Analyse MD trajectory and cluster into 10 clusters
    analysis.analyse(
        traj=trajectory,
        resname=residue,
        top=topology,
        RMSD=True,
        cluster=True,
        last=True,
        clust_type="BS",
        rmsd_type="BS",
        sieve=10,
        output_dir=output_dir,
    )

    # Check number of clusters
    clusters = glob.glob(os.path.join(output_dir, "cluster.*.pdb"))
    assert len(clusters) == 10

    # Check info.dat
    info = os.path.join(output_dir, "info.dat")
    errors = check_file(info, expected_line)
    assert not errors

    # Check plot
    plots = glob.glob(os.path.join(output_dir, "*.png"))
    assert plots

    # Clean up data
    check_remove_folder(output_dir)
