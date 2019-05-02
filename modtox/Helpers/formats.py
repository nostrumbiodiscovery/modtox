import os
import subprocess
import modtox.constants.constants as cs


def  pdb_to_mae(pdb, schr=cs.SCHR, output=None):
    if not output:
        output = os.path.splitext(os.path.basename(pdb))[0]+".mae"
    pdbconvert = os.path.join(schr, "utilities/pdbconvert")
    command = "{} -ipdb {}  -omae {}".format(pdbconvert, pdb, output)
    print(command)
    subprocess.call(command.split())
    return output

def  sd_to_mae(sdf, schr=cs.SCHR, output=None):
    if not output:
        output = os.path.splitext(os.path.basename(sdf))[0]+".mae"
    sdconvert = os.path.join(schr, "utilities/sdconvert")
    command = "{} -isdf {}  -omae {}".format(sdconvert, sdf, output)
    print(command)
    subprocess.call(command.split())
    return output

def convert_to_mae(list_of_files):
    ligands_to_dock_mae = []
    for ligand in list_of_files:
        extension = ligand.split(".")[-1]
        if extension == "pdb":
            ligand_mae = pdb_to_mae(ligand)
            ligands_to_dock_mae.append(ligand_mae)
        elif extension == "sdf":
            ligand_mae = sd_to_mae(ligand)
            ligands_to_dock_mae.append(ligand_mae)
        else:
            ligands_to_dock_mae.append(ligand)
    return ligands_to_dock_mae
