from rdkit import Chem
import os
from chembl_webresource_client.unichem import unichem_client as unichem

class DUDE():
    
    def __init__(self, chmbl_folder):
        self.actives = os.path.join(chmbl_folder, "actives_final.ism")
        self.decoys = os.path.join(chmbl_folder, "decoys_final.ism")
        self.actives_sdf = os.path.join(chmbl_folder, "actives_final.sdf")
        self.decoys_sdf = os.path.join(chmbl_folder, "decoys_final.sdf")
        
    def actives_to_chembl(self):
        with open(self.actives, "r") as f:
            for line in f:
                if line:
                    yield line.split()[-1] 
            
    def chembl_to_molecule(self, chembl):
        for name in chembl:
            try:
                yield Chem.MolFromInchi(unichem.structure(name,1)[0]["standardinchi"])
            except IndexError:
                print("Molecules {} not found".format(name))
        
    def molecules_to_sdf(self, molecules, output="output.sdf"):
        w = Chem.SDWriter(output)
        for m in molecules:
            w.write(m)
    
    def preprocess_ligands(sdf):
        pass


def parse_args(parser):
    parser.add_argument("--dataset",  type=str, help='DUD-E dataset folder')
    parser.add_argument("--output", type=str, help='sdf output', default="output.sdf")

def dude_set(dataset_folder="../cp3a4/", output="cyp_actives.sdf"):
    parser = argparse.ArgumentParser(description="Preprocess dataset from chembl by using ligprep over the actives having  \
    into account tautomerization, chirality and protonation of active ligands. Inactive are left as DUDE output them.", formatter_class=RawTextHelpFormatter)
    parse_args()
    args = parser.parse_args(parser)
    chmbl = ChemBL(args.dataset)
    chmbl_names = chmbl.actives_to_chembl()
    molecules = chmbl.chembl_to_molecule(chmbl_names)
    chmbl.molecules_to_sdf(molecules, args.output) 



if __name__ == "__main__":
    dataset_folder, output = parse_args()
    dude_set(dataset_folder, output)
