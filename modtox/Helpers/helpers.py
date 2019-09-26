import os

def retrieve_molecule_number(pdb, resname):

    """
     IDENTIFICATION OF MOLECULE NUMBER BASED
	ON THE TER'S
    """
    count = 0
    with open(pdb, 'r') as x:
        lines = x.readlines()
        for i in lines:
            if i.split()[0] == 'TER': count += 1
            if i.split()[3] == resname:             
                molecule_number = count + 1
                break
    return molecule_number



class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
