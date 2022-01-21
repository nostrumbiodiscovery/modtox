import os
import time
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

    def __init__(self, folder, active, inactive, systems, debug=False):
 
        self.folder = folder
        self.active = [active]
        self.inactive = [inactive]
        self.systems = systems
        self.greasy_template = greasy_template
        self.input_template = input_template
        self.runjob_greasy_template = runjob_greasy_template
        self.debug = debug

    def _format_checking(self):
        extension_receptor = self.systems[0].split(".")[-1]
        extension_active = self.active[0].split(".")[-1]
        extension_inactive = self.inactive[0].split(".")[-1]
        assert type(self.systems) == list, "receptor must be of list type"
        assert extension_active in ACCEPTED_FORMATS, "receptor must be a pdb, sdf or mae"
        assert extension_inactive in ACCEPTED_FORMATS, "ligand must be a pdb, sdf or mae at the moment"
        assert extension_receptor in ['zip'], "receptor must be zip type"

    def sorting(self, i, *args):
        print('Ordering non-sorted files', args)
        sorted_files = []
        for fil in args:
            newname2 = fil.split('.')[0] + '_sorted.' + fil.split('.')[1]
            command_sorting = "/opt/schrodinger2019-4/utilities/glide_sort  -o {} {}".format(newname2, fil) 
            print('from', fil, newname2)
            subprocess.call(command_sorting.split())
            time.sleep(3)
            sorted_files.append(newname2)
        command_greasy = "/opt/schrodinger2019-1/utilities/glide_merge -o {}/$new $old".format(self.folder)
        newname =  "input_merged_{}.maegz".format(int(i/2))
        oldname = "{} {}".format(sorted_files[0], sorted_files[1])
        command_greasy = command_greasy.replace("$old", oldname)
        command_greasy = command_greasy.replace("$new", newname)
        try:
            subprocess.check_output(command_greasy.split())
        except subprocess.CalledProcessError as e:
            print(e.output)
            assert False
        return command_greasy
        

    def merge_glides(self, inp_files):

      #"/opt/schrodinger2019-1/utilities/glide_merge ../docking/input5*.maegz > new.maegz"
        assert len(inp_files) == 20
        new_files = []
        if not self.debug:
            for i in range(0, len(inp_files), 2):
                issorted = True
                print('------>', i)
                command_greasy = "/opt/schrodinger2019-1/utilities/glide_merge $old -o {}/$new".format(self.folder)
                newname = "input_merged_{}.maegz".format(int(i/2))
                oldname = "{} {}".format(inp_files[i], inp_files[i+1])
                command_greasy = command_greasy.replace("$old", oldname)
                command_greasy = command_greasy.replace("$new", newname)
                new_files.append(os.path.join(self.folder, newname))
                try:
                    subprocess.check_output(command_greasy.split())
                except subprocess.CalledProcessError as e:
                    issorted = False
                    command_greasy = self.sorting(i, inp_files[i], inp_files[i+1])
                    print('sorted')
                if issorted:
                    subprocess.call(command_greasy.split())
        print(new_files)
        return new_files


    def preparation(self): 

        self._format_checking()
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

