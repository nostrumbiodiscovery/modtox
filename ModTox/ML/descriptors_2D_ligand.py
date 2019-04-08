#IMPORT MODULES AND FILE
import pandas as pd
from rdkit import Chem
from scipy import stats
from sklearn import linear_model
from rdkit.Chem import AllChem
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols
import rdkit.Chem.Descriptors as dc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
import rdkit.Chem.Crippen as cr
import mordred as md
from mordred import Calculator, descriptors
import numpy as np
import pandas as pd
import sklearn
from rdkit import DataStructs
from sklearn.decomposition import KernelPCA
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import os
from nltk import TweetTokenizer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

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

    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels): 
        return self.transform(self.fit(molecules))

    def transform(self, molecules):
        print("\tBuilding MACS Fingerprints")
        df = pd.DataFrame()
        molecules = molecules["molecules"].tolist()
        fingerprints = [MACCSkeys.GenMACCSKeys(mol).ToBitString() for mol in molecules]
        for i, fingerprint in enumerate(fingerprints):
            df = df.append(pd.Series({"rdkit_fingerprintMACS_{}".format(j):element for j, element in enumerate(fingerprint)}), ignore_index=True)
        np.savetxt("MAC_descriptors.txt", list(df), fmt="%s")
        return df.astype(float)
    
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

    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels): 
        return self.transform(self.fit(molecules))

    def transform(self, molecules):
        print("\tBuilding Daylight Fingerprints")
        df = pd.DataFrame()
        molecules = molecules["molecules"].tolist()
        fingerprints = [FingerprintMols.FingerprintMol(mol).ToBitString() for mol in molecules]
        for i, fingerprint in enumerate(fingerprints):
            df = df.append(pd.Series({"rdkit_fingerprint_{}".format(j):element for j, element in enumerate(fingerprint)}), ignore_index=True)   
        np.savetxt("daylight_descriptors.txt", list(df), fmt="%s")
        return df.astype(float)

class Descriptors():
    
    def __init__(self, features=None, headers=None):
        self.descriptors = features
        self.headers = headers
    
    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels): 
        return self.transform(self.fit(molecules))

    def transform(self, molecules):
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
        np.savetxt("2D_descriptors.txt", list(descriptors_df), fmt="%s")
        return  descriptors_df.astype(float)
    
class Descriptors_Schordinger():

    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels): 
        return self.transform(self.fit(molecules))

    def transform(self, molecules):
        print("\tBuilding Schrodinger Descriptors")
        df = pd.DataFrame()
        finger_prints_train = preprocessor.fit_transform(molecules)
        for i, mol in enumerate(molecules):
            desc = mol.GetPropsAsDict()
            for x in ENTITIES_TO_REMOVE:
                if x in desc:
                    del desc[x]
            df = df.append(pd.Series(desc), ignore_index=True)
        return df

class Shape():

    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels): 
        return self.transform(self.fit(molecules))

    def transform(self, molecules):
        df = pd.DataFrame()
        molecules = molecules["molecules"].tolist()
        df["Shape"] = [mol.GetPropsAsDict()["r_m_Shape_Sim"] if "r_m_Shape_Sim" in mol.GetPropsAsDict().keys() else None for mol in suppl]
        return df
    
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
