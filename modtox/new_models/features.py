"""
---------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
This script calculates other descriptors and merges them into balanced Glide features 
dataframe. 

Can be used as individual script or implemented in a pipeline. 
---------------------------------------------------------------------------------------------------------------------
USAGE (as script)
Contains two main functions, which can be called using different flags:
    
    1.  add_features(): adds mordred descriptors and/or topological fingerprints to balanced 
        Glide dataframe.
        Usage(*): python features.py [--csv glide_features.csv] [--mordred True/False] [--topological True/False]
            OUTPUT: balanced glide + descriptors CSV. 

    2.  all_combs():    generates 7 dataframes, for all different combinations of Glide features,
                        mordred descriptors and topological fingerprints. 
        Usage(*): python features.py [--csv glide_features.csv] --all
            OUTPUT: 7 CSV files, all combinations (glide.csv, mordred.csv, fingerprints.csv, glide_mordred.csv...)

    * Be aware that --all flag overwrites --topological and --mordred.
    * If --csv flag is not provided, it loads 'balanced_glide.csv' from current working 
    directory. Default options can be defined in '_parameters.py'. 
---------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
"""

import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from mordred import Calculator, descriptors
import os
import itertools

from modtox.modtox.utils import utils
from modtox.modtox.new_models._parameters import *

"""
--------------------------------------------------------------------------------------------
MAIN FUNCTIONS
--------------------------------------------------------------------------------------------
"""
def add_features(balanced_glide_csv=BALANCED_GLIDE, topological_fingerprints=True, mordred_descriptors=True, 
                                        active_sdf=ACTIVES_SDF, inactive_sdf=INACTIVES_SDF, savedir=SAVEDIR):
    """
    Adds one or both descriptors to balanced Glide features file.
    Default arguments defined in '_parameters.py'.
    -------------------------------------------------------------
    OUTPUT: CSV file with balanced Glide features + descriptors.
    RETURN: Dataframe with balanced Glide features + descriptors.
    """
    # Can't be chained to 'glide_processing.py' because set balancing is random. 
    # Instead, read from output of 'glide_processing.py'.
    balanced_glide, all_mols, all_mol_names = get_balanced_glide(balanced_glide_csv, active_sdf, inactive_sdf)

    dfs = ["glide"] # To use as file name.
    # eval() is mandatory because argparse passes strings, not bool.  
    # If set to False, descriptors are set to empty dataframe. 
    if eval(topological_fingerprints):
        fingerprints = calculate_fingerprints(all_mols, all_mol_names)
        dfs.append("fingerprints")
    else:
        fingerprints = pd.DataFrame()

    if eval(mordred_descriptors):
        mordred = calculate_mordred(all_mols, all_mol_names)
        dfs.append("mordred")
    else:
        mordred = pd.DataFrame()

    # Merges non-empty dataframes. 
    features = utils.merge_ifempty(mordred, fingerprints, MOL_NAME_COLUMN)
    glide_features = utils.merge_ifempty(balanced_glide, features, MOL_NAME_COLUMN)
    
    filename = "_".join(dfs)
    
    # Sorted and formatted for testing purposes.  
    glide_features = glide_features.sort_index()
    # Drops all columns before "Title" (indexing and others)
    glide_features = utils.drop_before_column(glide_features, MOL_NAME_COLUMN)

    utils.save_to_csv(glide_features, "Features dataframe", os.path.join(savedir, filename + ".csv"))
    
    return glide_features

def all_combs(combo, balanced_glide_csv=BALANCED_GLIDE, active_sdf=ACTIVES_SDF, inactive_sdf=INACTIVES_SDF, savedir=SAVEDIR):
    """
    Generates all possible features combinations (Glide, mordred and fingerprints)
    and stores in specified directory ($SAVEDIR/features), created if not exists. 
        1.  Read from balanced Glide. 
        2.  Calculate fingerprints and mordred.
        3.  Generate all combinations.
        4.  Merge the three dataframes.
        5.  Add 'Activity' column if not present and save to CSV
    -------------------------------------------------------------------------------
    INPUT:  combo: list of descriptors to combine
            'balanced_glide.csv'
    RETURN: Dictionary as {df_name: <pd.DataFrame>}
    OUTPUT: CSV files with the specified dataframes
    """
    # True/False for each descriptor
    is_glide = "glide" in combo
    is_mordred = "mordred" in combo
    is_fingerprints = "fingerprints" in combo

    # Import balanced glide
    if is_glide:
        balanced_glide_df, all_mols, all_mol_names = get_balanced_glide(balanced_glide_csv, 
                                                                        active_sdf, inactive_sdf)
    # Calculate descriptors
    if is_mordred:
        print("Calculating mordred descriptors...")
        mordred_df = calculate_mordred(all_mols, all_mol_names)
    
    if is_fingerprints:
        print("Calculating topological fingerprints...")    
        fingerprints_df = calculate_fingerprints(all_mols, all_mol_names)

    # Create 'features' directory if non-existent
    features_dir = os.path.join(savedir, "features")
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
        print(f"Features directory created at: '{features_dir}'")
    
    # Generate all possible combinations of the provided
    # and remove first element (empty)
    combs = list()
    for L in range(0, len(combo)+1):
        for subset in itertools.combinations(combo, L):
            combs.append(list(subset))
    del combs[0]

    # Empty dictionary to populate with pd.DataFrames
    d = {}
    
    # Iterate over combinations
    for comb in combs:
        # Set variables to dataframe for each loop.
        # copy() defines a new identical dataframe instead of referring.
        if is_glide: glide = balanced_glide_df.copy()
        if is_mordred: mordred = mordred_df.copy()
        if is_fingerprints: fingerprints = fingerprints_df.copy()

        # If name not in combination, set to empty dataframe
        if "glide" not in comb:
            glide = pd.DataFrame()
        if "mordred" not in comb:
            mordred = pd.DataFrame()
        if "fingerprints" not in comb:
            fingerprints = pd.DataFrame()        
        
        # Merge all in "Title"
        features = utils.merge_ifempty(mordred, fingerprints, MOL_NAME_COLUMN)
        final_df = utils.merge_ifempty(glide, features, MOL_NAME_COLUMN)
        
        # Add activity column if non-existent (if Glide dataframe was set 
        # to empty, no 'Activity' column is present)
        if not ACTIVITY_COLUMN in list(final_df.columns):
            molecule_names = final_df[MOL_NAME_COLUMN].astype(str).tolist()
            activity = ["/active" in element for element in molecule_names]
            final_df[ACTIVITY_COLUMN] = activity    

        df_name = ", ".join(comb)  # for printing
        file_name = "_".join(comb)  # for saving file
        
        # Sorted and formatted for testing purposes.
        final_df = final_df.sort_index()
        final_df = utils.drop_before_column(final_df, MOL_NAME_COLUMN)

        # Save dataframe to dictionary and CSV
        d[file_name] = final_df
        utils.save_to_csv(final_df, f"Features ({df_name})", os.path.join(features_dir, file_name + ".csv"))
    return d

"""
--------------------------------------------------------------------------------------------
HELPER FUNCTIONS
--------------------------------------------------------------------------------------------
"""
def get_balanced_glide(balanced_glide, active_sdf, inactive_sdf):
    """
    Retrieves all molecules and its names from the active/inactive 
    sdf file to match with the ones in balanced glide. It is necessary
    because balancing is random. 
    ----------------------------------------------------------------
    INP: balanced_glide.csv, actives/inactives sdf files.
    OUT: all_molecules <rdkit.obj> list and names list.
    """
    balanced_glide_df = pd.read_csv(balanced_glide)
    mols_in_bal_glide = balanced_glide_df[MOL_NAME_COLUMN].to_list()

    # Retrieve molecules from SDF files in dict to avoid duplicates if
    # are present in balanced glide dataframe. 
        # For enantiomers:  last molecule in SDF file is selected, same as 
        #                   docking (glide_features.csv)
    actives = {
        mol.GetProp("_Name"): mol 
        for mol in Chem.SDMolSupplier(active_sdf)
        if mol.GetProp("_Name") in mols_in_bal_glide
    }

    inactives = {
        mol.GetProp("_Name"): mol 
        for mol in Chem.SDMolSupplier(inactive_sdf)
        if mol.GetProp("_Name") in mols_in_bal_glide
    }

    actives.update(inactives)
    all_molecules_dict = actives

    all_molecules = all_molecules_dict.values()
    all_molecule_names = all_molecules_dict.keys()
    
    # Drop all columns until "Title", reindex, convert to float. 
    balanced_glide_df = utils.drop_before_column(balanced_glide_df, MOL_NAME_COLUMN)
    
    return balanced_glide_df, all_molecules, all_molecule_names

def calculate_fingerprints(all_molecules, all_molecule_names):
    """
    Calculates fingerprints of supplied list.
    -------------------------------------------------------------
    INP: <rdkit.obj> list for calculation
         all molecule names to add "Title" column
    OUT: Fingerprints dataframe (with "Title")
    """
    daylight_fingerprints = [Chem.RDKFingerprint(mol) for mol in all_molecules]
    fingerprints = pd.DataFrame()
    for i, fingerprint in tqdm(enumerate(daylight_fingerprints)):
        fingerprints = fingerprints.append(
            pd.Series(
                {
                    "rdkit_fingerprint_{}".format(j): int(element)
                    for j, element in enumerate(fingerprint)
                }
            ),
            ignore_index=True,
        )
    fingerprints.insert(0, "Title", all_molecule_names)

    return fingerprints

def calculate_mordred(all_molecules, all_molecule_names):
    """
    Calculates mordred descriptors of supplied list.
    -------------------------------------------------------------
    INP: <rdkit.obj> list for calculation
         all molecule names to add "Title" column
    OUT: Mordred descriptors dataframe (with "Title")
    """
    mordred = pd.DataFrame()
    mordred_descriptors = Calculator(descriptors, ignore_3D=True)
    mordred = pd.concat(
        [mordred, mordred_descriptors.pandas(mols=all_molecules)], axis=1
    )
    mordred.insert(0, "Title", all_molecule_names)

    return mordred

"""
--------------------------------------------------------------------------------------------
ARGPARSE
--------------------------------------------------------------------------------------------
"""
def parse_args(parser):
    parser.add_argument("--csv", help="Path to CSV file with Glide features. If flag not supplied, assumed to be in CWD as 'balanced_glide.csv'.")
    parser.add_argument("--topological", help="Calculate rdkit topological fingerprints - True or False.")
    parser.add_argument("--mordred", help="Calculate Mordred descriptors - True or False.")
    parser.add_argument("--all", help="Generates all combinations of Glide features, mordred descriptors and topological fingerprints."
                                        " Be aware that overwrites --topological and --mordred flags.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supply path to the CSV file with Glide features and decide which 2D descriptors should be calculated."
        " Add --all flag to obtain all possible combinations between Glide features, mordred descriptors and topological"
        f" fingerprints. Saved in CSV files in {SAVEDIR}/features."
    )
    parse_args(parser)
    args = parser.parse_args()
    if args.all:
        combo = ["glide", "mordred", "fingerprints"]
        all_combs(combo, balanced_glide_csv=args.csv)
    else:
        add_features(args.csv, args.topological, args.mordred)
