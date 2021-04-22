import os
import pandas as pd
import requests

from rdkit import Chem
from rdkit.Chem import rdmolops

chembl_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/{}.sdf"


class ChEMBL:

    def __init__(self, csv, threshold, folder_output="."):
        """
        Initializes ChEMBL processor extracts data from a standard ChEMBL CSV file.

        Parameters
        -----------
        csv : str
            Path to the CSV file downloaded from ChEMBL.
        threshold : float
            Threshold for separating actives and inactives based on Ki.
        folder_output : str, optional
            Folder to save the SD files with actives and inactives.
        """
        self.dataframe = self.load_csv(csv)
        self.threshold = threshold
        self.folder_output = folder_output
        self.id_column = "Molecule ChEMBL ID"
        self.activity_column = "Standard Value"
        self.actives_df, self.inactives_df = None, None

    def get_data(self):
        """
        Runs the whole pipeline:
        - filters out duplicates based on ChEMBL IDs and SMILES
        - splits them into actives and inactives based on the user-define threshold
        - downloads SD files for each molecule
        - assigns stereochemistry and sanitizes the molecules.

        Returns
        --------
        actives : str
            Path to SD file with active compounds.
        inactives : str
            Path to SD file with inactive compounds.
        """
        if not os.path.exists(self.folder_output):
            os.mkdir(self.folder_output)

        self.actives_df, self.inactives_df = self.preprocess()
        actives_file = self.download_sdf(df=self.actives_df, file_name="actives")
        inactives_file = self.download_sdf(df=self.inactives_df, file_name="inactives")

        return actives_file, inactives_file

    def preprocess(self):
        """
        Preprocess the dataframe:
        - remove compounds without associated activity
        - remove duplicates based on ChEMBL ID and SMILEs
        - split into actives and inactives dataframes based on threshold

        Returns
        --------
        df_actives : pandas.Dataframe
            ChEMBL dataframe with active compounds only (Ki >= threshold)
        df_inactives : pandas.Dataframe
            ChEMBL dataframe with inactive compounds only (Ki < threshold)
        """
        df = self.dataframe
        df = df.drop(df[df[self.activity_column].isnull()].index)  # drop NaNs in the activity column
        df = df.drop_duplicates(subset=[self.id_column, "Smiles"])
        df_actives = df[df[self.activity_column] < self.threshold]
        df_inactives = df[df[self.activity_column] >= self.threshold]

        self.save_csv(self.folder_output, "actives.csv", df_actives)
        self.save_csv(self.folder_output, "inactives.csv", df_inactives)

        print(f"Identified {len(df_actives)} active and {len(df_inactives)} inactive compounds.")

        return df_actives, df_inactives

    def download_sdf(self, df, file_name):
        """
        Downloads SD molecules form ChEMBL based on a dataframe and saves them to file.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing ChEMBL IDs in "Molecule ChEMBL ID" column.
        file_name : str
            Name of the SD file to be created.
        """
        print(f"Downloading {file_name}...")

        failed_downloads = []

        ids = df[self.id_column].tolist()
        sdf_content = list()

        for chembl_id in ids:
            url = chembl_url.format(chembl_id.strip())
            response = requests.get(url)

            if "no structure records" not in response.text:
                sdf_content.append(response.text + "\n$$$$\n")
            else:
                failed_downloads.append(chembl_id)

        if failed_downloads:
            print(f"Failed to download: {', '.join(failed_downloads)}.")
        file_name = os.path.join(self.folder_output, f"{file_name}.sdf")

        with open(file_name, "w+") as file:
            for line in sdf_content:
                file.write(line)

        sanitized_file = self.structure_cleanup(file)
        return sanitized_file

    @staticmethod
    def load_csv(csv):
        """
        Reads in the CSV file as pandas DataFrame.

        Parameters
        -----------
        csv : str
            Path to ChEMBL CSV file.

        Returns
        --------
        dataframe : pandas.DataFrame
            Dataframe from CSV file.
        """
        dataframe = pd.read_csv(csv, delimiter=";")
        print(f"Loading {csv}.")
        return dataframe

    @staticmethod
    def save_csv(file_name, folder, dataframe):
        """
        Saves dataframe to a CSV file.

        Parameters
        -----------
        file_name : str
            File name for the CSV to be created.
        folder : str
            Path to the output folder
        dataframe : pandas.DataFrame
            Dataframe object to be exported to CSV.
        """
        path = os.path.join(file_name, folder)
        print(f"Saving {path}.")
        dataframe.to_csv(path)

    @staticmethod
    def structure_cleanup(file):
        """
        Reads in created SD files, sanitizes the molecules and assigns stereochemistry.
        """
        supplier = Chem.SDMolSupplier(file.name)
        output_file = "{}_sanitized.sdf".format(os.path.splitext(file.name)[0])

        for molecule in supplier:
            rdmolops.AssignStereochemistry(molecule, force=True, cleanIt=True)
            Chem.SanitizeMol(molecule)

        writer = Chem.SDWriter(output_file)
        for molecule in supplier:
            writer.write(molecule)
        print(f"Saved sanitized molecules to {output_file}.")
        return output_file
