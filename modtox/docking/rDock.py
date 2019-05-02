from string import Template
import argparse
import os
from argparse import RawTextHelpFormatter
import subprocess

DIR = os.path.dirname(__file__)


class rDocker(object):
    
    def __init__(self, receptor, ligand_to_dock, ligand_for_grid):
        self.receptor = receptor
        self.ligand_to_dock = ligand_to_dock
        self.ligand_for_grid = ligand_for_grid

    def dock(self, grid="grid.prm", output="output"):
        # Set grid, output and commands
        self.grid = os.path.abspath(os.path.join(DIR, grid))
        self.output = output
        self.grid_command = "rbcavity -was -d -r {}".format(grid)
        self.docking_command = "rbdock -i {} -o {} -r {} -p dock.prm -n 50".format(self.ligand_to_dock, self.output, grid)
        
        # Templetize grid
        with open(self.grid, "r") as f:
            template = Template("\n".join(f.readlines()))
            content = template.safe_substitute(RECEPTOR=self.receptor, LIGAND=self.ligand_for_grid)
        with open(grid, "w") as fout:
            fout.write(content)

        # Build cavity
        print(self.grid_command)
        subprocess.call(self.grid_command.split())

        # Run docking
        print(self.docking_command)
        subprocess.call(self.docking_command.split())

def parse_args():
    parser = argparse.ArgumentParser(description='Specify Receptor and ligand to be docked\n  \
    i.e python -m modtox.docking.dock receptor ligand_to_dock --grid ligand_for_grid', formatter_class=RawTextHelpFormatter)
    parser.add_argument('receptor', type=str, help='Receptor to build the grid for docking on')
    parser.add_argument('ligands_to_dock', type=str, help='sdf file for ligands to be docked')
    parser.add_argument('--grid', type=str, help='sdf file with a single ligand to build from')
    args = parser.parse_args()
    return args.receptor, args.ligands_to_dock, args.grid

if __name__ == "__main__":
    receptor, ligands_to_dock, ligand_for_grid = parse_args()
    docking_obj = RDocker(receptor, ligands_to_dock, ligand_for_grid) 
    docking_obj.dock()
