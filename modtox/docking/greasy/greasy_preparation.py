import os
import glob
import modtox.Helpers.formats as fm
import subprocess
from string import Template
from tqdm import tqdm


ACCEPTED_FORMATS = ["pdb", "mae", "sdf", "maegz"]
GRID = "GRIDFILE {}"
LIGAND = "LIGANDFILE {}"
greasy_template = '/home/moruiz/modtox_dir/modtox/modtox/docking/greasy/greasy_template.txt'
input_template = '/home/moruiz/modtox_dir/modtox/modtox/docking/greasy/input_template.txt'
runjob_greasy_template = '/home/moruiz/modtox_dir/modtox/modtox/docking/greasy/runjob_greasy_template.sh'

class GreasyObj():

    def __init__(self, folder, active, inactive, systems):
 
        self.folder = folder
        self.active = [active]
        self.inactive = [inactive]
        self.systems = systems
        self.greasy_template = greasy_template
        self.input_template = input_template
        self.runjob_greasy_template = runjob_greasy_template
        self._format_checking()

    def _format_checking(self):
        extension_receptor = self.systems[0].split(".")[-1]
        extension_active = self.active[0].split(".")[-1]
        extension_inactive = self.inactive[0].split(".")[-1]
        assert type(self.systems) == list, "receptor must be of list type"
        assert extension_receptor in ACCEPTED_FORMATS, "receptor must be a pdb, sdf or mae"
        assert extension_receptor in ACCEPTED_FORMATS, "ligand must be a pdb, sdf or mae at the moment"

    def preparation(self): 

        self.systems_mae = fm.convert_to_mae(self.systems, folder=self.folder)
        self.active_to_dock_mae = fm.convert_to_mae(self.active, folder=self.folder)
        self.inactive_to_dock_mae = fm.convert_to_mae(self.inactive, folder=self.folder)

        gridfile = [GRID.format(os.path.basename(system)) for system in self.systems_mae]
        ligandfile = [LIGAND.format(os.path.basename(ligand)) for ligand in self.active_to_dock_mae + self.inactive_to_dock_mae]

        # Templetize input
        self.input_files = []
        for j,i in enumerate(tqdm(gridfile)):
            for k,l in enumerate(ligandfile):
                with open(self.input_template, "r") as f:
                    template = Template("".join(f.readlines()))
                    content = template.safe_substitute(GRIDFILE="{}".format(i), LIGANDFILE="{}".format(l))
                input_file = os.path.basename(self.input_template).split("_template")[0] + "{}{}.in".format(j,k)
                with open(os.path.join(self.folder, input_file), "w") as fout:
                    fout.write(content)
                self.input_files.append(input_file)
       
        # Templetize greasy 
        with open(self.greasy_template, "r") as f:
            template = Template("".join(f.readlines()))
            content = [template.safe_substitute(input_file="{}".format(inp)) for inp in self.input_files]
        self.greasy_file = os.path.basename(self.greasy_template).split("_template")[0] + ".txt"
        with open(os.path.join(self.folder, self.greasy_file), "w") as fout:
            fout.write(''.join(content))

        # Templetize runjob
        with open(self.runjob_greasy_template , "r") as f:
            template = Template("".join(f.readlines()))
            content = template.substitute(ntasks="1", greasy_file="{}".format(self.greasy_file)) #ntasks to be modified when more inputs
        self.runjob_file = os.path.basename(self.runjob_greasy_template).split("_template")[0] + ".sh"
        with open(os.path.join(self.folder, self.runjob_file), "w") as fout:
            fout.write(''.join(content))
 
        print('Greasy templetize done!')

