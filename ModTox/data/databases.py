import requests
from rdkit.Chem import AllChem
from rdkit import Chem
from tqdm import tqdm
from chembl_webresource_client.unichem import unichem_client as unichem
from chembl_webresource_client.new_client import new_client



class PullDB():

    def __init__(self, ids, source="chembl"):
        self.ids = ids
        self.source = source
        if source == "zinc":
            self.inchi_keys = self.ids
        elif source == "chembl":
            self.inchi_keys = self._from_chembl(self.ids)
        self.n_compounds = len(ids)
        self.__zinc_url__ = " http://zinc15.docking.org/substances/{}.sdf"
        self.__chmbl_url__ =  "https://www.ebi.ac.uk/chembl/api/data/molecule/{}.sdf" 

    def _from_chembl(self, ids):
        for name in ids:
            for struct in unichem.structure(name,1):
                 yield str(struct["standardinchi"])

    def to_sdf(self, output="active.sdf"):

        molecules = []
        molecules_rdkit = []
        w = Chem.SDWriter(output)
        for inchy, name in tqdm(zip(self.inchi_keys, self.ids), total=self.n_compounds-1):
            try:
                m = Chem.inchi.MolFromInchi(inchy, removeHs=True)
                Chem.AssignStereochemistry(m)
                AllChem.EmbedMolecule(m)
                m.SetProp("_Name", name)
                m.SetProp("_MolFileChiralFlag", "1")
                molecules_rdkit.append(m)
            except IndexError:
                print("Molecules {} not found".format(name))

        for m in molecules_rdkit: w.write(m)

        return output, len(molecules_rdkit)


if __name__ == "__main__":
    zinc_obj = PullDB(["CHEMBL520419", ], source="chembl")
    zinc_obj.to_sdf()
