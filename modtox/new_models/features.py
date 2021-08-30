"""
Contains two main functions:
    1.  add_features(): processes Glide features ('glide_processing.py') and 
                        adds mordred descriptors and/or topological fingerprints. 
        Usage: python features.py --csv glide_features.csv --mordred True/False --fingerprints True/False
    -------------------------------------------------------------------------------------    
    2.  all_combs():    generates dataframes for all combinations between Glide 
                        features, mordred descriptors and topological fingerprints.
        Usage: python features.py --csv glide_features.csv --all
    -------------------------------------------------------------------------------------
"""

import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from mordred import Calculator, descriptors
from math import floor
import os
import itertools

from modtox.utils import utils as u

MOL_NAME_COLUMN = "Title"
ACTIVITY_COLUMN = 'Activity'
ACTIVES_SDF = 'actives.sdf'
INACTIVES_SDF = 'inactives.sdf'
EXTERNAL_SET_PROP = 0.05


def add_features(balanced_glide_csv, topological_fingerprints=True, mordred_descriptors=True, 
                                        active_sdf=ACTIVES_SDF, inactive_sdf=INACTIVES_SDF, savedir=os.getcwd()):
    """
    Adds extra features to balanced Glide features csv. 
        1.  Glide CSV processing (glide_processing.py).
        2.  Evaluate arguments and calculate extra descriptors.
            If not calculated, set to empty dataframe. 
        3.  Merge the three dataframes.
        4.  Save to .csv file. 
    -------------------------------------------------------------
    INPUT:  'glide_features.csv', finger/mordred = T/F
    OUTPUT: Glide dataframe with the specified features added. 
    """
    balanced_glide, all_mols, all_mol_names = get_balanced_glide(balanced_glide_csv, active_sdf, inactive_sdf)

    dfs = ["glide"] # To use as file name. 
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

    features = u.merge_ifempty(mordred, fingerprints, "Title")
    glide_features = u.merge_ifempty(balanced_glide, features, "Title")
    
    filename = "_".join(dfs)
    
    glide_features = glide_features.sort_index()
    glide_features = u.drop_before_column(glide_features, "Title")
    u.save_to_csv(glide_features, "Features dataframe", os.path.join(savedir, filename))
    
    return glide_features

def all_combs(balanced_glide_csv, combo, active_sdf=ACTIVES_SDF, inactive_sdf=INACTIVES_SDF, savedir=os.getcwd()):
    """
    Generates all possible features combinations (Glide, mordred and fingerprints)
    and stores in specified directory ($FEATURES_DIR), created if not exists. 
        1.  Glide CSV processing (glide_processing.py).
        2.  Calculate fingerprints and mordred.
        3.  Generate all combinations.
        4.  Add 'Activity' column if not in df.columns
        5.  Merge the three dataframes and save to CSV. 
    -------------------------------------------------------------------------------
    INPUT:  'glide_features.csv'
    OUTPUT: Dictionary as df_name: <pd.DataFrame> and CSV files
    """
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

    # Create directory if non-existent
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
        # Set variables to dataframe for each loop
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
        
        # Merge all
        features = u.merge_ifempty(mordred, fingerprints, "Title")
        final_df = u.merge_ifempty(glide, features, "Title")
        
        # Add activity column if non-existent (if Glide dataframe was set 
        # to empty, no 'Activity' column is present)
        if not ACTIVITY_COLUMN in list(final_df.columns):
            molecule_names = final_df[MOL_NAME_COLUMN].astype(str).tolist()
            activity = ["/active" in element for element in molecule_names]
            final_df[ACTIVITY_COLUMN] = activity    

        df_name = ", ".join(comb)  # for printing
        file_name = "_".join(comb)
        
        final_df = final_df.sort_index()
        
        final_df = u.drop_before_column(final_df, "Title")

        # Save dataframe to dictionary and CSV
        d[file_name] = final_df
        u.save_to_csv(final_df, f"Features ({df_name})", os.path.join(features_dir, file_name))
    return d

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

    # Retrieve molecules from SDF files in dict to avoid duplicates.
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
    
    # Drop all columns until "Title", reindex, convert to float and fill NaN
    balanced_glide_df = u.drop_before_column(balanced_glide_df, MOL_NAME_COLUMN)
    
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
    Calculating fingerprints and mordred descriptors duplicates 
    (no idea why).
    """
    mols = df


def parse_args(parser):
    parser.add_argument("--csv", help="Path to CSV file with Glide features.")
    parser.add_argument("--topological", help="Calculate rdkit topological fingerprints - True or False.")
    parser.add_argument("--mordred", help="Calculate Mordred descriptors - True or False.")
    parser.add_argument("--all", help="Generates all combinations of Glide features, mordred descriptors and topological fingerprints.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supply path to the CSV file with Glide features and decide which 2D descriptors should be calculated."
    )
    parse_args(parser)
    args = parser.parse_args()
    if args.all:
        all_combs(args.csv)
    else:
        add_features(args.csv, args.topological, args.mordred)
