import pandas as pd
import numpy as np



class ExternalData():

    def __init__(self, csv, mol_names, exclude=[]):
        self.csv = csv
        self.mol_names = mol_names       
        self.exclude = exclude

    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels): 
        return self.transform(self.fit(molecules))

    def transform(self, molecules):
        print("\tIncorporating external data")
        df = pd.DataFrame.from_csv(self.csv)
        n_drop = 0
        #If not present on your dataset discard
        self.mol_names = [m[0].GetProp("_Name") for m in molecules.values]
        for i, title in enumerate(df["Title"].values):
            if title not in self.mol_names:
                df.drop(df.index[i- n_drop], inplace=True)
                n_drop += 1
        headers = list(df)
        values = (None,) * len(headers)
        #If not present on your glide discard
        for i, title in enumerate(self.mol_names):
            if title not in df["Title"].values:
                line = pd.DataFrame.from_records([values], columns=headers, index=[i])
                df = pd.concat([df.ix[:i-1], line, df.ix[i:]]).reset_index(drop=True)
        #Drop features
        df = df.replace("--",  np.nan)
        features_to_drop = [feature for field in self.exclude for feature in headers if field in feature ]
        df.drop(features_to_drop, axis=1, inplace=True)
        df.to_csv("model_features.txt")
        return df

    def retrieve_molecule_names(self):
        df = pd.DataFrame.from_csv(self.csv) 
        thresh = int(df.shape[1]*0.8)
        df_dropna = df.dropna(thresh=thresh)
        return df_dropna["Title"].values 
