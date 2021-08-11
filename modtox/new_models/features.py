import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from mordred import Calculator, descriptors


MOL_NAME_COLUMN = "Title"
ACTIVITY_COLUMN = 'Activity'


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
    glide = pd.read_csv(csv)
    glide.sort_values(MOL_NAME_COLUMN, inplace=True)
    docked_molecule_names = glide[MOL_NAME_COLUMN].astype(str).tolist()

    # Get actives and inactives labels (True/False)
    activity = ["/active" in element for element in docked_molecule_names]
    glide[ACTIVITY_COLUMN] = activity

    # Load molecules from SD files, if they were docked (i.e. appear in the CSV Title column) and non-repeating
    print("Reading in the SD files.")
    checked_molecules = list()
    actives = [mol for mol in Chem.SDMolSupplier("actives.sdf") if mol]
    inactives = [mol for mol in Chem.SDMolSupplier("inactives.sdf") if mol]

    docked_actives = [
        mol
        for mol in actives
        if mol.GetProp("_Name") in docked_molecule_names
        and mol.GetProp("_Name") not in checked_molecules
    ]

    skipped_actives = [
        mol
        for mol in actives
        if mol.GetProp("_Name") not in docked_molecule_names
        and mol.GetProp("_Name") not in checked_molecules
    ]

    docked_inactives = [
        mol
        for mol in inactives
        if mol.GetProp("_Name") in docked_molecule_names
        and mol.GetProp("_Name") not in checked_molecules
    ]

    skipped_inactives = [
        mol
        for mol in inactives
        if mol.GetProp("_Name") not in docked_molecule_names
        and mol.GetProp("_Name") not in checked_molecules
    ]

    print(
        f"Retrieved {len(skipped_actives + docked_actives)} active "
        f"and {len(skipped_inactives + docked_inactives)} inactive molecules."
    )

    updated_glide = insert_zero_rows(skipped_actives, skipped_inactives, glide)
    updated_glide.to_csv("updated_glide_features.csv")

    return actives, inactives, glide


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
    for mol in skipped_inactives + skipped_actives:

        if ":" not in mol.GetProp("_Name"):
            continue  # skipping the ones without molecule name

        df_dict = {key: [0] for key in dataframe.columns}
        df_dict[ACTIVITY_COLUMN] = [mol in skipped_actives]
        df_dict[MOL_NAME_COLUMN] = [mol.GetProp("_Name")]

        row_df = pd.DataFrame.from_dict(df_dict)
        dataframe = pd.concat([dataframe, row_df])

    return dataframe


def generate_features(actives, inactives, topological_fingerprints, mordred_descriptors):
    all_molecules = actives + inactives
    all_molecule_names = [mol.GetProp("_Name") for mol in all_molecules]

    if mordred_descriptors:
        print("Calculating Mordred descriptors...")
        mordred = pd.DataFrame()
        mordred_descriptors = Calculator(descriptors, ignore_3D=True)
        mordred = pd.concat(
            [mordred, mordred_descriptors.pandas(mols=all_molecules)], axis=1
        )
        print("Mordred", mordred.shape)
        mordred.to_csv("mordred.csv")
    else:
        mordred = pd.DataFrame()

    if topological_fingerprints:
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
        print("Fingerprints", fingerprints.shape)
        fingerprints.to_csv("fingerprints.csv")
    else:
        fingerprints = pd.DataFrame()

    # Merging based on molecule index
    features = pd.merge(
        fingerprints, mordred, "outer", left_index=True, right_index=True
    )

    features.insert(0, "ID", all_molecule_names)
    print("Merged", features.shape)
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
    final_df = pd.merge(features, glide, "inner", left_on=["ID"], right_on=["Title"])
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
