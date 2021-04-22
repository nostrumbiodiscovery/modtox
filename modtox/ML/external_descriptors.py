import pandas as pd
import numpy as np
import os


class ExternalData:

    def __init__(self, csv, mol_names, folder='.', exclude=[]):
        self.csv = csv
        self.mol_names = mol_names
        self.exclude = exclude
        self.folder = folder
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

    def fit(self, molecules):
        return molecules

    def fit_transform(self, molecules, labels):
        return self.transform(self.fit(molecules))

    def retrieve_mols_not_docked(self, molecules):
        df = pd.read_csv(self.csv)
        molecules_not_docked = []
        headers = list(df)
        values = (None,) * len(headers)
        for i, title in enumerate(self.mol_names):
            if title not in df["Title"].values.astype(str):
                null_line = pd.DataFrame.from_records([values], columns=headers, index=[i])
                df = pd.concat([df.iloc[:i-1], null_line, df.iloc[i:]]).reset_index(drop=True)
                molecules_not_docked.append(title)
        return molecules_not_docked

    def transform(self, molecules):
        ''' important to remember: df follows the order of the csv whereas mol_names the order of the sdf files'''
        print("\tIncorporating external data from {}".format(self.csv))
        df = pd.read_csv(self.csv)

        # If any of the SDF actives/inactives is not in the CSV, discard it:
        n_drop = 0
        self.mol_names = [m[0].GetProp("_Name") for m in molecules.values]  # 228, looks like rdkit mols from X

        for i, title in enumerate(df["Title"].values):
            if str(title) not in self.mol_names:
                df.drop(df.index[i - n_drop], inplace=True)
                n_drop += 1

        # Check whether molecules in mol_names not in df (i.e. actives/inactives that were not docked)
        headers = list(df)
        values = (None,) * len(headers)

        # If not present on your glide
        print('Filling null lines...')
        for i, title in enumerate(self.mol_names):
            if title not in df["Title"].values.astype(str):
                null_line = pd.DataFrame.from_records([values], columns=headers, index=[i])
                df = pd.concat([df.loc[:i - 1], null_line, df.loc[i:]]).reset_index(drop=True)

        # Drop features (columns)
        df = df.replace("--", np.nan)
        df['Title'] = df.Title.astype(str)
        df = df.sort_values("Title")
        features_to_drop = [feature for field in self.exclude for feature in headers if field in feature]
        df.drop(features_to_drop, axis=1, inplace=True)
        df.to_csv(os.path.join(self.folder, "model_features.txt"))

        return df

    def retrieve_molecule_names(self):
        df = pd.read_csv(self.csv)
        thresh = int(df.shape[1] * 0.8)
        df_dropna = df.dropna(thresh=thresh)
        return df_dropna["Title"].values
