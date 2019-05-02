import modtox.constants.constants as cs
from string import Template
import subprocess
import os
import time


DIR = os.path.dirname(__file__)

def ligprep(sdf, template="ligprep.in", input_file="ligprep.in", schr=cs.SCHR, output="ligands_proc.mae"):
    template = os.path.join(DIR, template)
    ligprep_bin = os.path.join(schr, "ligprep")
    command = "{} -inp {}".format(ligprep_bin, input_file)

    # Templetize grid
    with open(template, "r") as f:
        template = Template("".join(f.readlines()))
        content = template.safe_substitute(LIGANDS=sdf, OUTPUT=output)
    with open(input_file, "w") as fout:
        fout.write(content)

    print(command)
    subprocess.call(command.split())

    while not os.path.exists(output):
        time.sleep(60)


    return os.path.abspath(output)

    
