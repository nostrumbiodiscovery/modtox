from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols
from mordred import Calculator, descriptors
import numpy as np
import pandas as pd
from rdkit import DataStructs
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def fpList_to_bit(fp_list):
    return DataStructs.CreateFromBitString("".join(fp_list))


#CREATE dataframe and molecules
class Similarity():
    def fit(self, fingerprints):
        return fingerprints

    def fit_transform(self, fingerprints): 
        return self.transform(self.fit(fingerprints))

    def transform(self, fingerprints):
        print("\tBuilding Similarity Components")
        similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)))
        for i, fp_1 in enumerate(fingerprints):
            fp_1 = fpList_to_bit(fp_1)
            for j, fp_2 in enumerate(fingerprints):
                fp_2 = fpList_to_bit(fp_2)
                similarity_matrix[i][j] = DataStructs.FingerprintSimilarity(fp_1,fp_2, metric=DataStructs.DiceSimilarity)
        #transformer = KernelPCA(n_components=7, kernel='linear')
        #X_transformed = transformer.fit_transform(similarity_matrix)
        return pd.DataFrame(similarity_matrix)
    
class Similarity_decomp():
    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels): 
        return self.transform(self.fit(molecules))

    def transform(self, molecules):
        print("\tBuilding Similarity Components")
        molecules = molecules["molecules"].tolist()
        fingerprints = [FingerprintMols.FingerprintMol(mol).ToBitString() for mol in molecules]
        return Similarity().fit_transform(fingerprints)

class Fingerprints_MACS():

    def __init__(self, folder='.'): 
        self.folder = folder
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels, folder='.'): 
        return self.transform(self.fit(molecules), folder)

    def transform(self, molecules, folder = '.'):

        print("\tBuilding MACS Fingerprints")
        df = pd.DataFrame()
        molecules = molecules["molecules"].tolist()
        MACCS = [MACCSkeys.GenMACCSKeys(mol) for mol in molecules]
        fingerprints = []
        for mac in tqdm(MACCS):
            fingerprints.append([int(bit) for bit in mac])
        for i, fingerprint in tqdm(enumerate(fingerprints)):
            df = df.append(pd.Series({"rdkit_fingerprintMACS_{}".format(j):element for j, element in enumerate(fingerprint)}), ignore_index=True)

        np.savetxt(os.path.join(self.folder, "MAC_descriptors.txt"), list(df), fmt="%s")
        return df
    
class Fingerprints_Morgan():

    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels): 
        return self.transform(self.fit(molecules))

    def transform(self, molecules):
        print("\tBuilding Morgan Fingerprints")
        df = pd.DataFrame()
        molecules = molecules["molecules"].tolist()
        fingerprints = [AllChem.GetMorganFingerprint(mol,2).GetTotalVal() for mol in molecules]
        return pd.DataFrame(fingerprints)
    
class Fingerprints():

    def __init__(self, folder='.'): 
        self.folder = folder
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)


    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels, folder='.'): 
        return self.transform(self.fit(molecules), folder)

    def transform(self, molecules, folder='.'):
        print("\tBuilding Daylight Fingerprints")
        df = pd.DataFrame()
        molecules = molecules["molecules"].tolist()
        fingerprints = [Chem.RDKFingerprint(mol) for mol in molecules]
        for i, fingerprint in tqdm(enumerate(fingerprints)):
            df = df.append(pd.Series({"rdkit_fingerprint_{}".format(j): int(element) for j, element in enumerate(fingerprint)}), ignore_index=True)   
        np.savetxt(os.path.join(self.folder, "daylight_descriptors.txt"), list(df), fmt="%s")
        return df.astype(float)

class Descriptors():
    
    def __init__(self, folder = '.', features=None, headers=None):
        self.descriptors = features
        self.headers = headers
        self.folder = folder 
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels, folder='.'): 
        return self.transform(self.fit(molecules), folder)

    def transform(self, molecules, folder='.'):
        print("\tBuilding Descriptors")
        df = pd.DataFrame()
        molecules = molecules["molecules"].tolist()
        #df["MW"] = [dc.FpDensityMorgan1(mol) for mol in molecules]
        if self.descriptors:
            print(self.descriptors)
            calcs = Calculator(self.descriptors, ignore_3D=True) 
        else:
            calcs = Calculator(descriptors, ignore_3D=True)
        #calcs = Calculator([md.CarbonTypes, md.LogS, md.ABCIndex, md.BondCount, md.ZagrebIndex, md.WienerIndex,md.TopologicalCharge, md.InformationContent, md.AcidBase,md.RingCount, md.AtomCount, md.Polarizability, md.HydrogenBond,md.SLogP,md.RotatableBond, md.Aromatic, md.CPSA], ignore_3D=True) 
        #df["MG"] = [dc.FpDensityMorgan1(mol) for mol in molecules]
        #df["headers"] = list(df)*(df.shape[0]+1)
        descriptors_df = pd.concat([df, calcs.pandas(molecules)], axis=1)
        if self.headers:
            descriptors_df["headers"] = [list(descriptors_df)]*descriptors_df.shape[0]
        np.savetxt(os.path.join(self.folder, "2D_descriptors.txt"), list(descriptors_df), fmt="%s")
        return  descriptors_df.astype(float)
    

class Smiles():
    
    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels): 
        return self.transform(self.fit(molecules))

    def transform(self, molecules):
        print("\tBuilding Smiles")
        df = pd.DataFrame()
        molecules = molecules["molecules"].tolist()
        smiles = [ Chem.MolToSmiles(mol, isomericSmiles=False) for mol in molecules]
        vectorizer = TfidfVectorizer(lowercase=False, analyzer='char', ngram_range=(1, 4), min_df=1)
        return vectorizer.fit_transform(smiles)
