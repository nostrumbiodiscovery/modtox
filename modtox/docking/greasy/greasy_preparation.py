import os
import glob
import modtox.Helpers.formats as fm
import subprocess
from string import Template
from tqdm import tqdm


ACCEPTED_FORMATS = ["pdb", "mae", "sdf", "maegz"]
GRID = "GRIDFILE {}"
LIGAND = "LIGANDFILE {}"
DIR = os.path.dirname(os.path.abspath(__file__))
greasy_template = os.path.join(DIR, 'greasy_template.txt')
input_template = os.path.join(DIR, 'input_template.txt')
runjob_greasy_template = os.path.join(DIR, 'runjob_greasy_template.sh')

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
        assert extension_active in ACCEPTED_FORMATS, "receptor must be a pdb, sdf or mae"
        assert extension_inactive in ACCEPTED_FORMATS, "ligand must be a pdb, sdf or mae at the moment"
        assert extension_receptor in ['zip'], "receptor must be zip type"

    def preparation(self): 

        self.active_to_dock_mae = fm.convert_to_mae(self.active, folder=self.folder)
        self.inactive_to_dock_mae = fm.convert_to_mae(self.inactive, folder=self.folder)

        gridfile = [GRID.format(os.path.basename(system)) for system in self.systems]
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
            content = template.substitute(ntasks= len(gridfile)*len(ligandfile), greasy_file="{}".format(self.greasy_file)) #ntasks to be modified when more inputs
        self.runjob_file = os.path.basename(self.runjob_greasy_template).split("_template")[0] + ".sh"
        with open(os.path.join(self.folder, self.runjob_file), "w") as fout:
            fout.write(''.join(content))
 
        print('Greasy templetize done!')

