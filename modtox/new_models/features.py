import argparse
import pandas as pd
from pandas.core.reshape.merge import merge
from tqdm import tqdm
from rdkit import Chem
from mordred import Calculator, descriptors
import random

MOL_NAME_COLUMN = "Title"
ACTIVITY_COLUMN = 'Activity'
ACTIVES_SDF = 'actives.sdf'
INACTIVES_SDF = 'inactives.sdf'


def run(csv, topological_fingerprints=True, mordred_descriptors=True):
    """
    Run the whole flow of extracting Glide features, calculating fingerprints and descriptors, etc.

    Parameters
    ----------
    csv : str
        Path to CSV file with Glide features.
    topological_fingerprints : bool
        Toggle to calculate topological fingerprints from rdkit.
    mordred_descriptors : bool
        Toggle to calculate Mordred descriptors.

    Returns
    -------
        Dataframe with final model input.
    """

    actives, inactives, glide = retrieve_molecules(csv)
    features = generate_features(actives, inactives, topological_fingerprints, mordred_descriptors)
    model_input = merge_all(features, glide)
    return model_input


def retrieve_molecules(csv):
    """
    Retrieves molecules from the Glide features CSV file and compares with the original actives and inactives SD files.

    Parameters
    ----------
    csv : str
        Path to CSV file with Glide features.

    Returns
    -------
        SDMolSupplier with actives, inactives and updated Glide features CSV.
    """
    # Get docked molecules (true and false positives)
    glide = pd.read_csv(csv)
    glide.sort_values(MOL_NAME_COLUMN, inplace=True)
    docked_molecule_names = glide[MOL_NAME_COLUMN].astype(str).tolist()

    # Tag true positives (True) and false positives (False) in the glide.DataFrame
    #                                                 from name e.g.: "sdf/active_sanitzed.sdf:38"
    activity = ["/active" in element for element in docked_molecule_names]
    glide[ACTIVITY_COLUMN] = activity

    # Get all molecules from SD files (true positives and true negatives) 
    # as a dictionary to remove repeated molecules -> 'CHEMBL02914': (<rdkit.obj>, _Name) 
    print("Reading in the SD files.")
    actives = {
        mol.GetProp("chembl_id"): (mol, mol.GetProp("_Name")) 
        for mol in Chem.SDMolSupplier(ACTIVES_SDF)
        if ":" in mol.GetProp("_Name")  
    }

    inactives = {
        mol.GetProp("chembl_id"): (mol, mol.GetProp("_Name")) 
        for mol in Chem.SDMolSupplier(INACTIVES_SDF)
        if ":" in mol.GetProp("_Name") 
    }
    
    # Iterate over dictionary keys, append <rdkit.obj> if _Name... is/not in docked_molecule_names
        # actives[chmblid] = (<rdkit.obj>, _Name)
            # actives[chmblid][0] = <rdkit.obj>
            # actives[chmblid][1] = _Name
    
    # True positives
    docked_actives = [
        actives[chmblid][0]  
        for chmblid in actives.keys()
        if actives[chmblid][1] in docked_molecule_names and
           ":" in actives[chmblid][1]
    ]

    # False negatives
    skipped_actives = [
        actives[chmblid][0]
        for chmblid in actives.keys()
        if actives[chmblid][1] not in docked_molecule_names and
           ":" in actives[chmblid][1]        
    ]

    # False positives
    docked_inactives = [
        inactives[chmblid][0]
        for chmblid in inactives.keys()
        if inactives[chmblid][1] in docked_molecule_names and
           ":" in inactives[chmblid][1]
    ]

    # True negatives
    skipped_inactives = [
        inactives[chmblid][0]
        for chmblid in inactives.keys()
        if inactives[chmblid][1] not in docked_molecule_names and
           ":" in inactives[chmblid][1]
    ]
    
    print(
        f"Retrieved {len(skipped_actives + docked_actives)} active " 
        f"and {len(skipped_inactives + docked_inactives)} inactive molecules."
    )

    # From dictionary to <rdkit.obj> list
    actives = [actives[key][0] for key in actives.keys()]
    inactives = [inactives[key][0] for key in inactives.keys()]
    
    # Balance sets
    print("Balancing sets...")
    while len(actives) != len(inactives):
        if len(actives) > len(inactives):
            random_index = random.randint(0, len(actives) - 1)
            del actives[random_index]
        if len(actives) < len(inactives):
            random_index = random.randint(0, len(inactives) - 1)
            del inactives[random_index]
    
    print(f"Sets balanced. {len(actives)} active molecules and {len(inactives)} inactives.")
    
    updated_glide = insert_zero_rows(skipped_actives, skipped_inactives, glide)
    updated_glide.to_csv("updated_glide_features.csv")

    return actives, inactives, updated_glide


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

def generate_features(actives, inactives, topological_fingerprints, mordred_descriptors):
    
    all_molecules = actives + inactives
    all_molecules = [
        mol
        for mol in all_molecules
        if ":" in mol.GetProp("_Name")
    ]

    all_molecule_names = [
        mol.GetProp("_Name") 
        for mol in all_molecules
        if ":" in mol.GetProp("_Name")
    ]

    if eval(mordred_descriptors):
        print("Calculating Mordred descriptors...")
        mordred = pd.DataFrame()
        mordred_descriptors = Calculator(descriptors, ignore_3D=True)
        mordred = pd.concat(
            [mordred, mordred_descriptors.pandas(mols=all_molecules)], axis=1
        )
        mordred.insert(0, "Title", all_molecule_names)
        print("Mordred", mordred.shape)
        mordred.to_csv("mordred.csv")
    else:
        mordred = pd.DataFrame()

    if eval(topological_fingerprints):
        print("Calculating fingerprints...")
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
        print("Fingerprints", fingerprints.shape)
        fingerprints.to_csv("fingerprints.csv")
    else:
        fingerprints = pd.DataFrame()
    
    features = merge_features(mordred, fingerprints)
    return features

def merge_features(mordred, fingerprints):
    '''
    Merges mordred and fingerprints, checking for empty dataframes. 
    '''
    try:
        # Try to merge both
        features = pd.merge(mordred, fingerprints, on="Title")
    except:
        # If both empty, return empty DataFrame
        if mordred.empty and fingerprints.empty:
            features = pd.DataFrame()
        # Else if, return the only populated DataFrame
        elif mordred.empty:    
            features = fingerprints
        elif fingerprints.empty:
            features = mordred
    return features

def merge_all(features, glide):
    """
    Merges calculated features with Glide input and returns a CSV file with model input.

    Parameters
    ----------
    features : pd.DataFrame
        Dataframe containing calculated features such as fingerprints and descriptors.
    glide : pd.DataFrame
        Dataframe containing Glide features.

    Returns
    -------
        Dataframe with the final model input.
    """
    print("Creating model input...")


    file = "model_input.csv"

    if features.empty:
        final_df = glide
    else:
        final_df = pd.merge(glide, features, on="Title")
    
    final_df.to_csv(file)
    print(f"Saved to {file}!")

    return final_df


def parse_args(parser):
    parser.add_argument("--csv", help="Path to CSV file with Glide features.")
    parser.add_argument("--topological", help="Calculate rdkit topological fingerprints - True or False.")
    parser.add_argument("--mordred", help="Calculate Mordred descriptors - True or False.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supply path to the CSV file with Glide features and decide which 2D descriptors should be calculated."
    )
    parse_args(parser)
    args = parser.parse_args()
    run(args.csv, args.topological, args.mordred)
