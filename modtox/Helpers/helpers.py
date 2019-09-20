

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
