from string import Template
import argparse
import os
from argparse import RawTextHelpFormatter
import subprocess
import modtox.Helpers.formats as fm
import modtox.constants.constants as cs

DIR = os.path.dirname(__file__)
COMPLEX_LINE = "COMPLEX   {},{}"
LIGAND_LINE = "LIGAND   {}"
ACCEPTED_FORMATS = ["pdb", "mae", "sdf"]

class Glide_Docker(object):
    
    def __init__(self, systems, ligands_to_dock, test=False):
        self.systems = systems
        self.ligands_to_dock = ligands_to_dock
        self.test = test

    def dock(self, input_file="input.in", schr=cs.SCHR, host="localhost:1", cpus=1, output="glide_output", precision="SP",
        maxkeep=500, maxref=40, grid_mol=2):

        # Security type check
        extension_receptor = self.systems[0].split(".")[-1]
        extension_ligand = self.ligands_to_dock[0].split(".")[-1]
        assert type(self.systems) == list, "receptor must be of list type"
        assert type(self.ligands_to_dock) == list, "ligand file must be of list type"
        assert extension_receptor in ACCEPTED_FORMATS, "receptor must be a pdb, sdf or mae"
        assert extension_receptor in ACCEPTED_FORMATS, "ligand must be a pdb, sdf or mae at the moment"

        #Formats
        self.systems_mae = fm.convert_to_mae(self.systems)
        self.ligands_to_dock_mae = fm.convert_to_mae(self.ligands_to_dock)

        # Set dock command
        self.docking_command = '{}run xglide.py {} -OVERWRITE -HOST {} -NJOBS {} -TMPLAUNCHDIR -ATTACHED'.format(
        schr, input_file, host, cpus)

        # Set variables for docking        
        self.grid_template = os.path.abspath(os.path.join(DIR, input_file))
        complexes = [COMPLEX_LINE.format(system, grid_mol) for system in self.systems_mae]
        ligands = [LIGAND_LINE.format(ligand) for ligand in self.ligands_to_dock_mae]

        # Templetize grid
        with open(self.grid_template, "r") as f:
            template = Template("".join(f.readlines()))
            content = template.safe_substitute(COMPLEXES="\n".join(complexes), LIGANDS="\n".join(ligands),
         PRECISION=precision, MAXKEEP=maxkeep, MAXREF=maxref)
        with open(input_file, "w") as fout:
            fout.write(content)

        # Run docking
        print(self.docking_command)
        if not self.test:
            subprocess.call(self.docking_command.split())



def parse_args(parser):
    parser.add_argument('--receptor', "-r",  nargs="+", help='Receptor to build the grid for docking on')
    parser.add_argument('--ligands_to_dock', "-l", nargs="+", help='sdf file for ligands to be docked')
    parser.add_argument('--grid', type=str, help='sdf file with a single ligand to build from')
    parser.add_argument('--precision', type=str, help='Docking precision [SP (default), XP]', default="SP")
    parser.add_argument('--maxkeep', type=int, help='Maximum number of initial poses (Major speed up)', default=500)
    parser.add_argument('--maxref', type=str, help='Maximum number fo poses to keep at the end of each iteration', default=40)
    parser.add_argument('--grid_mol', type=int, help='Number of the molecule to be used as grid. (Each TER is a molecule)', default=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify Receptor and ligand to be docked\n  \
    i.e python -m modtox.docking.dock receptor ligand_to_dock --grid ligand_for_grid', formatter_class=RawTextHelpFormatter)
    parse_args(parser)
    args = parser.parse_args()
    docking_obj = Glide_Docker(args.receptor, args.ligands_to_dock) 
    docking_obj.dock(precision=args.precision, maxkeep=args.maxkeep, maxref=args.maxref)
