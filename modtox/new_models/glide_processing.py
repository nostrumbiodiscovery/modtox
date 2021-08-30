"""
This script processes the output from Glide
--------------------------------------------------------------------------------------------
INPUT  -> glide_features.csv
OUTPUT -> actives/inactives: lists of <rdkit.obj> with the molecules in the training dataset.
       -> balanced_glide_df: balanced dataframe (actives/inactives) with Glide features. 
                             Undocked molecules added and rows set to 0. 
--------------------------------------------------------------------------------------------
USAGE:
"""

import argparse
from unicodedata import name
import pandas as pd
from rdkit import Chem
import random
import os

from modtox.utils import utils as u


MOL_NAME_COLUMN = "Title"
ACTIVITY_COLUMN = 'Activity'
ACTIVES_SDF = 'actives.sdf'
INACTIVES_SDF = 'inactives.sdf'
EXTERNAL_SET_PROP = 0.05

def run(csv, active_sdf, inactive_sdf, savedir=os.getcwd()):
    """
    1. Reads Glide features file into .csv
    2. Adds undocked molecules to the dataframe.
    3. Balances sets. 
    4. Inserts '0' in NaN cells.
    -----------------------------------------------------------
    INP:    csv file with Glide features
    OUT:    list of 'actives' and 'inactives' <rdkit.obj> and 
            'balanced_glide' dataframe
    """
    dataframe, docked_molecule_names, activity = tag_activity(csv)
    actives, inactives, docked_actives, docked_inactives, skipped_actives, skipped_inactives = read_sdf(docked_molecule_names, active_sdf, inactive_sdf)

    print(
        f"Retrieved {len(skipped_actives + docked_actives)} active " 
        f"and {len(skipped_inactives + docked_inactives)} inactive molecules."
    )
    print("Balancing sets...", end="")
    balance_sets(actives, inactives)
    print(f"Balanced. {len(actives)} active/inactive molecules.")

    all_molecules_glide = insert_zero_rows(skipped_actives, skipped_inactives, dataframe)
    all_molecules_glide.reset_index(inplace=True)

    balanced_glide_df = balance_glide(all_molecules_glide, actives, inactives)
    balanced_glide_df.sort_index(inplace=True)


    # Drop all columns until "Title", reindex, convert to float and reindex
    balanced_glide_df = u.drop_before_column(balanced_glide_df, MOL_NAME_COLUMN)

    
    u.save_to_csv(balanced_glide_df, "Balanced Glide", os.path.join(savedir, "balanced_glide"))
    return actives, inactives, balanced_glide_df


def tag_activity(csv):
    """
    1. Reads Glide csv (tp and fp)
    2. Tags T/F in 'Activity' column according to name (sdf/active_sanitzed.sdf:38)
    ---------------------------------------------------------------------------------
    INP:    csv with Glide features
    OUT:    tagged dataframe, list of docked molecule names and activity. 
    """

     # Get docked molecules (true and false positives)
    dataframe = pd.read_csv(csv)
    dataframe.sort_values(MOL_NAME_COLUMN, inplace=True)
    docked_molecule_names = dataframe[MOL_NAME_COLUMN].astype(str).tolist()

    # Tag true positives (True) and false positives (False) in the glide.DataFrame
    #                                                 from name e.g.: "sdf/active_sanitzed.sdf:38"
    activity = ["/active" in element for element in docked_molecule_names]
    dataframe[ACTIVITY_COLUMN] = activity
    return dataframe, docked_molecule_names, activity

def read_sdf(docked_molecule_names, active_sdf, inactive_sdf):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Current SDF files contain duplicate names (enantiomers).
    Those can be renamed checking if the name is in the list
    and then renaming with mol.SetProp("_Name", oldname_1) 
    
    Problem is glide_features.csv does NOT contain data about
    which enantiomer is docked.
    -----------------------------------------------------------
    1. Reads all molecules from sdf.
    2. Compares to docked_molecules and generates tp tn fp fn
    -----------------------------------------------------------
    INP:    list of docked molecules, active/inactive sdf files.
    OUT:    actives/inactives -> list of rdkit objects
            docked_actives, docked_inactives, skipped_actives, skipped_inactives
    """
    # Get all molecules from SD files (true positives and true negatives) 
    # as a dictionary to remove repeated molecules. Delete molecules with 
    # empty name field. 
    
    actives = {
        mol.GetProp("_Name"): mol 
        for mol in Chem.SDMolSupplier(active_sdf)
        if ":" in mol.GetProp("_Name")
    }

    inactives = {
        mol.GetProp("_Name"): mol 
        for mol in Chem.SDMolSupplier(inactive_sdf)
        if ":" in mol.GetProp("_Name")
    }
    
    # Iterate over dictionary keys, append <rdkit.obj> if _Name... is/not in docked_molecule_names
    # Data structure: actives[_Name] = <rdkit.obj>
    # True positives
    docked_actives = [
        actives[name]  
        for name in actives.keys()
        if name in docked_molecule_names
    ]

    # False negatives
    skipped_actives = [
        actives[name]
        for name in actives.keys()
        if name not in docked_molecule_names        
    ]

    # False positives
    docked_inactives = [
        inactives[name]
        for name in inactives.keys()
        if name in docked_molecule_names
    ]

    # True negatives
    skipped_inactives = [
        inactives[name]
        for name in inactives.keys()
        if name not in docked_molecule_names
    ]
    
    # From dictionary to <rdkit.obj> list
    actives = list(actives.values())
    inactives = list(inactives.values())
    
    return actives, inactives, docked_actives, docked_inactives, skipped_actives, skipped_inactives

def balance_sets(actives, inactives):
    """
    Removes by random index items from the greater list lenght
    from the OBJECT LISTS (actives/inactives), not in dataframe.
    ----------------------------------------------------------------
    INP:    actives/inactives list (lists all molecules from sdf files)
    OUT:    does not return, lists are modified in place. 
    """
    while len(actives) != len(inactives):
        if len(actives) > len(inactives):
            random_index = random.randint(0, len(actives) - 1)
            del actives[random_index]
        if len(actives) < len(inactives):
            random_index = random.randint(0, len(inactives) - 1)
            del inactives[random_index]
    return

def insert_zero_rows(skipped_actives, skipped_inactives, dataframe):
    """
    Inserts mock rows to the Glide CSV for the compounds that could not be docked.

    Parameters
    ----------
    skipped_actives : List
        List of active, undocked rdkit molecules.
    skipped_inactives : List
        List of inactive, undocked rdkit molecules.
    dataframe : pd.DataFrame
        Dataframe to insert the rows.

    Returns
    -------
        Original dataframe with inserted zero rows.
    """

    # Create new DataFrame to append to existing (glide)
    cols = [col for col in dataframe.columns]
    row_df = pd.DataFrame(columns=cols)

    # Iterate over <rdkit.obj>
    for mol in skipped_inactives + skipped_actives:

        # if ":" not in mol.GetProp("_Name"):
        #     continue  # skipping the ones without molecule name

        df_dict = {key: 0 for key in dataframe.columns}
        df_dict[ACTIVITY_COLUMN] = mol in skipped_actives
        df_dict[MOL_NAME_COLUMN] = mol.GetProp("_Name")
        row_df = row_df.append(df_dict, ignore_index=True)

    dataframe = dataframe.append(row_df)
    return dataframe

def balance_glide(all_molecules_glide, actives, inactives):
    """
    Removes rows with molecules not in actives/inactives list. 
    -------------------------------------------------------------------
    INP:    all_molecules_glide -> dataframe with all molecules and Glide 
            features (rows w/ 0 in undocked ones)
            actives/inactives contains the definitive list of molecules 
            for building the model.  
    OUT:    
    """
    # Obtain lists of all names in actives/inactives and external set
    actives_names = [ mol.GetProp("_Name") for mol in actives ]
    inactives_names = [ mol.GetProp("_Name") for mol in inactives ]
    all_names = actives_names + inactives_names
    
    # Iterate over row and index. Use row["Title"] to compare and index
    # to drop row. 
    for index, row in all_molecules_glide.iterrows():
        if not row["Title"] in all_names:
            all_molecules_glide.drop([index], inplace=True)
    balanced_glide = all_molecules_glide
    
    return balanced_glide

def parse_args(parser):
    parser.add_argument("--csv", help="Path to CSV file with Glide features.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supply path to the CSV file with Glide features."
    )
    parse_args(parser)
    args = parser.parse_args()
    run(args.csv)
