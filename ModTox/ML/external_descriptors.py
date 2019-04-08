import pandas as pd
import numpy as np



class ExternalData():

    def __init__(self, csv, mol_names, exclude=[]):
	self.csv = csv
        self.mol_names = mol_names       
        self.exclude = exclude

    def fit(self, csv):
        return self.csv

    def fit_transform(self, csv, labels): 
        return self.transform(self.fit(self.csv))

    def transform(self, csv):
        print("\tIncorporating external data")
        df = pd.DataFrame.from_csv(self.csv)
        n_drop = 0
        for i, title in enumerate(df["Title"].values):
            if title not in self.mol_names:
                df.drop(df.index[i- n_drop], inplace=True)
                n_drop += 1
        headers = list(df)
        values = (None,) * len(headers)
        for i, title in enumerate(self.mol_names):
            if title not in df["Title"].values:
                line = pd.DataFrame.from_records([values], columns=headers, index=[i])
                df = pd.concat([df.ix[:i-1], line, df.ix[i:]]).reset_index(drop=True)
        # Drop colum with high number of NaN
        df = df.replace("--",  np.nan)
        df.dropna(thresh=df.shape[0]/2, axis=1)
        #Drop features
        features_to_drop = [feature for field in self.exclude for feature in headers if field in feature ]
        df.drop(features_to_drop, axis=1, inplace=True)
        return df

    def retrieve_molecule_names(self):
        df = pd.DataFrame.from_csv(self.csv) 
        thresh = int(df.shape[1]*0.8)
        df_dropna = df.dropna(thresh=thresh)
        return df_dropna["Title"].values 
