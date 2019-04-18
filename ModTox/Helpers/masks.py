import prody as pd
import argparse



def parse_args():

    parser = argparse.ArgumentParser(description='Specify trajectory and topology to be analised')
    parser.add_argument('pdb', type=str, help='PDB to extract the mask from')
    parser.add_argument('resname', type=str, help='Resname of the residue to look for closest atoms')
    parser.add_argument('--radius', type=int, help='Cutof to look for closest residues', default=1)
    args = parser.parse_args()
    return args.pdb, args.resname, args.radius


def retrieve_closest(pdb, resname, radius=5):

    atoms = pd.parsePDB(pdb)
    try:
        print("within {} of resname {} and protein".format(radius, resname))
        residues = atoms.select("within {} of resname {} and protein".format(radius, resname))
    except AttributeError:
        raise AttributeError("{} not found".format(pdb))
    try:
        resnumbers = residues.getResnums()
    except AttributeError:
        raise AttributeError("Check ligand resname. Must conceed with the extracted last_sanapshot.pdb resname")
    resnumbers = [str(resnum) for resnum in set(resnumbers)] #We need a list of strings to join later
    mask = ":{}".format(",".join(resnumbers))
    print(mask)


if __name__ == "__main__":
    pdb, resname, radius = parse_args()
    retrieve_closest(pdb, resname, radius)





