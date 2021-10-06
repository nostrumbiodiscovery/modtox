"""
--------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------
This script processes the output from Glide:
    1. Reads Glide features file from CSV, e.g.: the docked molecules (tp and fp).
    2. Adds undocked molecules to the dataframe, e.g.: tn and fn
    3. Balances sets deleting from the larger.
    4. Inserts '0' in NaN cells, mainly the undocked molecules (no Glide features info).
Can be used as individual script or implemented in a pipeline. 
--------------------------------------------------------------------------------------------
USAGE (as script)*
python glide_processing.py [--csv glide_features.csv] [-a actives.sdf] [-i inactives.sdf]
    OUTPUT: balanced_glide.csv in current directory. 

    * If --csv, -a or -i flags are not provided, it loads 'balanced_glide.csv', 'actives.sdf' 
    and 'inactives.sdf' from current working directory. Default options can be defined in 
    '_parameters.py'. 
--------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------
"""
import argparse
import pandas as pd
from rdkit import Chem
import random
import os

from modtox.modtox.utils import utils
from modtox.modtox.new_models._parameters import *

"""
--------------------------------------------------------------------------------------------
MAIN FUNCTION
--------------------------------------------------------------------------------------------
"""
def process_glide(csv=GLIDE_FEATURES, active_sdf=ACTIVES_SDF, inactive_sdf=INACTIVES_SDF, savedir=SAVEDIR):
    """
    Function called when script executed. Default arguments
    defined in '_parameters.py'.
    -----------------------------------------------------------
    OUTPUT: CSV file with balanced Glide features.
    RETURN: active and inactive <rdkit.obj> lists and balanced 
            Glide features as pandas dataframe.
    """
    dataframe, docked_molecule_names, activity = tag_activity(csv)
    actives, inactives, docked_actives, docked_inactives, skipped_actives, skipped_inactives = read_sdf(docked_molecule_names, active_sdf, inactive_sdf)

    print(
        f"Retrieved {len(skipped_actives + docked_actives)} active " 
        f"and {len(skipped_inactives + docked_inactives)} inactive molecules."
    )

    balance_sets(actives, inactives)
    print(f"Sets balanced: {len(actives)} active/inactive molecules.")

    all_molecules_glide = insert_zero_rows(skipped_actives, skipped_inactives, dataframe)
    all_molecules_glide.reset_index(inplace=True)

    balanced_glide_df = balance_glide(all_molecules_glide, actives, inactives)
    balanced_glide_df.sort_index(inplace=True)

    # Drop all columns until "Title", reindex, convert to float and reindex
    balanced_glide_df = utils.drop_before_column(balanced_glide_df, MOL_NAME_COLUMN)

    utils.save_to_csv(balanced_glide_df, "Balanced Glide", os.path.join(savedir, BALANCED_GLIDE))
    return actives, inactives, balanced_glide_df

"""
--------------------------------------------------------------------------------------------
HELPER FUNCTIONS
--------------------------------------------------------------------------------------------
"""
def tag_activity(csv):
    """
    1. Reads Glide CSV into pandas Dataframe
    2. Tags T/F in new 'Activity' column according to name (e.g.: sdf/active_sanitzed.sdf:38)
    ---------------------------------------------------------------------------------
    INPUT:  path to CSV with Glide features.
    RETURN: Activity-tagged dataframe, list of docked molecule names and activity. 
    """
     # Get docked molecules (true and false positives)
    dataframe = pd.read_csv(csv)
    dataframe.sort_values(MOL_NAME_COLUMN, inplace=True)
    docked_molecule_names = dataframe[MOL_NAME_COLUMN].astype(str).tolist()

    # Tag true positives (True) and false positives (False) in the glide.DataFrame
                                                    # from name e.g.: "sdf/active_sanitzed.sdf:38"
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
    INPUT:  list of docked molecules, active/inactive sdf files.
    RETURN: actives/inactives -> list of rdkit objects
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
    INPUT:  actives/inactives list (lists all molecules from sdf files)
    RETURN: does not return, lists are modified in place. 
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
    INPUT:  all_molecules_glide -> dataframe with all molecules and Glide 
            features (rows w/ 0 in undocked ones)
            actives/inactives contains the definitive list of molecules 
            for building the model.  
    RETURN: Glide features dataframe with the same number of actives and
            inactives.     
    """
    # Obtain lists of all names in actives/inactives
    actives_names = [ mol.GetProp("_Name") for mol in actives ]
    inactives_names = [ mol.GetProp("_Name") for mol in inactives ]
    all_names = actives_names + inactives_names
    
    # Iterate over row and index. Use row["Title"] to compare and index
    # to drop row, if not present in list. 
    for index, row in all_molecules_glide.iterrows():
        if not row[MOL_NAME_COLUMN] in all_names:
            all_molecules_glide.drop([index], inplace=True)
    balanced_glide = all_molecules_glide
    
    return balanced_glide

"""
--------------------------------------------------------------------------------------------
ARGPARSE
--------------------------------------------------------------------------------------------
"""

def parse_args(parser):
    parser.add_argument("--csv", help="Path to CSV file with Glide features. If flag not supplied, assumed to be in CWD as 'glide_features.csv'.")
    parser.add_argument("-a", "--actives", help="Path to SDF file of active molecules. If flag not supplied, assumed to be in CWD as 'actives.sdf'.")
    parser.add_argument("-i", "--inactives", help="Path to SDF file of inactive molecules. If flag not supplied, assumed to be in CWD as 'inactives.sdf'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process the Glide features dataframe supplying CSV " 
        "and SDF files. If flags are not defined, files are assumed to be " 
        "in current working directory as: 'glide_features.csv', 'actives.sdf' "
        "and 'inactives.sdf'."
    )
    parse_args(parser)
    args = parser.parse_args()
    csv = args.csv
    a = args.actives
    i = args.inactives

    if not csv: csv = GLIDE_FEATURES
    if not a: a = ACTIVES_SDF
    if not i: i = INACTIVES_SDF

    process_glide(csv, a, i)
