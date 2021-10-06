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
import os

from modtox.modtox.new_models.glide_processing import process_glide
from modtox.modtox.new_models import features
from modtox.modtox.new_models import single_model
from modtox.modtox.utils import utils

from modtox.modtox.new_models._parameters import *

MODELS = ["knn", "lr", "svc", "tree", "nb"]
DESCRIPTORS = ["glide", "mordred", "fingerprints"]

"""
--------------------------------------------------------------------------------------------
MAIN FUNCTION
--------------------------------------------------------------------------------------------
"""
def score_models(glide_features, descriptors_inp, models_inp, active_sdf=ACTIVES_SDF, 
                                            inactive_sdf=INACTIVES_SDF, ext_prop=EXTERNAL_PROPORTION, savedir=SAVEDIR):
    """Builds all the combinations of the specified descriptors and 
    fits each combination to all the specified models. The output is
    a CSV file sorted by descending accuracy of each feature combination
    and model. 

    Each feature combination in split into external validation set (EVS) 
    and internal data set (IDS). The IDS is then split into training (TDS) 
    and testing data sets (tDS)

    Each TDS is fit to all the user-specified models. tDS and EVS are tested
    against the generated model. Finally, the IDS is also scored using a 
    cross-validation strategy (k=5). 

          Initial data set
                |
        ------------------
        |                |
        5%               95%
       EVS              IDS ------------- cross-validation 
        |                |                         |
        |           ------------                   |
    Accuracy        |          |                CV score
    external       70%        30%
                   TDS        tDS
                    |          |
                 Training   testing
                               |
                               |
                            Accuracy
                              test  
    Parameters
    ----------
    glide_features : str
        Path to CSV file unprocessed 'glide_features.csv' 
    
    descriptors_inp : str
        Initials of the selected descriptors, including Glide 
        features.
            e.g.: for Glide features and mordred desc -> gm
            "all" is interpreted as "gmd"
    
    models_inp : str
        Initials of the selected models, "all" or "none". 
    
    active_sdf : str ['actives.sdf']
        Path to the active molecules SDF file. Default is defined
        in '_parameters.py'.  
    
    inactive_sdf : str ['inactives.sdf']
        Path to the active molecules SDF file. Default is defined
        in '_parameters.py'. 
    
    ext_prop : float [0.05]
        Proportion of data extracted as external data set.

    savedir : str [CWD]
        Path to saving directory.
    
    Returns
    -------
    scoring_df : pandas dataframe
        Resulting dataframe gathering all scoring information for each
        features and model combination, columns being:
            "Glide_features" (T/F)
            "Mordred_descriptors" (T/F)
            "Topological_fingerprints" (T/F)
            "Model"
            "Accuracy_test"
            "Confusion_matrix_test"
            "Accuracy_external"
            "Confusion_matrix_external"
            "Cross-validation_score" (mean +/- std)

    Default usage
    -------------
    With all the input files in the current working directory:

        CWD/
        |-- actives.sdf
        |-- inactives.sdf
        '-- glide_features.csv

    A standard usage would be: python multiple_models.py -d all -m all ; the expected output being:

        CWD/
        |--features/
        |     |-- glide.csv
        |     |-- glide_mordred.csv
        |     |-- glide_fingerprints.csv
        |     |-- glide_mordred_fingerprints.csv
        |     |-- mordred.csv
        |     |-- fingerprints.csv
        |     `-- mordred_fingerprints.csv
        |-- balanced_glide.csv
        |-- scoring.csv
        |-- actives.sdf
        |-- inactives.sdf
        `-- glide_features.csv

    - 'features/' directory contains all the combinations of features used as an input for the model. 
    - 'balanced_glide.csv' is the processed Glide features dataframe. 
    - 'scoring.csv' is a summary of all the models with all the features combinations scored and sorted by descending accuracy. 

    """
    # Glide processing to balanced sets Glide
    process_glide(glide_features, active_sdf, inactive_sdf, savedir=savedir)
    balanced_glide_csv = os.path.join(savedir, BALANCED_GLIDE)

    # Descriptors and model parsing, from initials to list: gm -> ["glide", "mordred"])
    sel_descriptors = retrieve_descriptors(descriptors_inp)
    sel_models = retrieve_models(models_inp)
    
    # Saves all inputs for model fitting and obtains dataframe dictionary
    features_df_dict = features.all_combs(sel_descriptors, balanced_glide_csv, active_sdf, inactive_sdf, 
                                                                                                savedir=savedir)

    # Create empty scoring dataframe
    cols = [
        "Glide_features",
        "Mordred_descriptors",
        "Topological_fingerprints",
        "Model",
        "Accuracy_test",
        "Confusion_matrix_test",
        "Accuracy_external",
        "Confusion_matrix_external",
        "Cross-validation_score"
    ]
    scoring_df = pd.DataFrame(columns=cols)

    # Iterate over model inputs
    for df_name in features_df_dict.keys():
        df = features_df_dict[df_name] 
        
        print(f"For {df_name}...")
        df = single_model.load_model_input(df)
        # Extract 5% external validation, label and imputation
        main_df, external_df, X, y, X_ext, y_ext = single_model.extract_external_val(df, ext_prop)
        
        # Iterate over models
        for model in sel_models:
            # Wrapping function, see for details
            model, acc_test, conf_test, acc_ext, conf_ext, cv_score = do_everything(model, X, y, X_ext, y_ext)

            model_str = str(model)[0:str(model).index("(")] # Print until first parenthesis
            
            print_scores(model_str, acc_test, conf_test, acc_ext, conf_ext, cv_score)
            # Dictionary to append to dataframe
            d = {
                "Glide_features": "glide" in df_name,
                "Mordred_descriptors": "mordred" in df_name,
                "Topological_fingerprints": "fingerprints" in df_name,
                "Model": model_str,
                "Accuracy_test": acc_test,
                "Confusion_matrix_test": conf_test,
                "Accuracy_external": acc_ext,
                "Confusion_matrix_external": conf_ext,
                "Cross-validation_score": cv_score
            }

            scoring_df = scoring_df.append(d, ignore_index=True)

    # Convert to numeric, sort by 'Accuracy_test' and round to 2 decimals
    scoring_df["Accuracy_test"] = pd.to_numeric(scoring_df["Accuracy_test"], errors="coerce", downcast="float")        
    scoring_df = scoring_df.sort_values("Accuracy_test", ascending=False)
    scoring_df = scoring_df.round(decimals=2)
    utils.save_to_csv(scoring_df, "Descriptors/models score", os.path.join(savedir, SCORING))
    return scoring_df

"""
--------------------------------------------------------------------------------------------
HELPER FUNCTIONS
--------------------------------------------------------------------------------------------
"""
def retrieve_models(models_input):
    """Returns a list of models from the parsed argument.
        e.g.:   "all" -> ["knn", "lr", "svc", "tree", "nb"]
                "kst" -> ["knn", "svc", "tree"]
    
    Parameters
    ----------
    models_input : str
        Initials of models or "all".

    Returns
    -------
    selected_models : list of strings
        List of models.
    """
    d = {model[0]: model for model in MODELS}
    if models_input == "all":
        selected_models = MODELS
    else:
        selected_models = [ d[init] for init in d.keys() if init in models_input ]
    
    return selected_models

def retrieve_descriptors(descriptors_input):
    """Returns a list of descriptors from the parsed argument.
        e.g.:   "all" ->  ["glide", "mordred", "fingerprints"]
                "gf" ->   ["glide", "fingerprints"]
    
    Parameters
    ----------
    descriptors_input : str
        Initials of descriptors or "all".

    Returns
    -------
    selected_descriptors : list of strings
        List of descriptors.
    """

    d = {desc[0]: desc for desc in DESCRIPTORS}
    if descriptors_input == "all":
        selected_descriptors = DESCRIPTORS
    else:
        selected_descriptors = [ d[init] for init in d.keys() if init in descriptors_input ]

    return selected_descriptors

def print_scores(model_str, acc_test, conf_test, acc_ext, conf_ext, cv_score):
    print(f"\tFor {model_str}:")
    print(f"\t\tInternal test accuracy: {acc_test}")
    print(f"\t\tInternal test confusion matrix: {conf_test}")
    print(f"\t\tExternal validation accuracy: {acc_ext}")
    print(f"\t\tExternal validation confusion matrix: {conf_ext}")
    print(f"\t\tCross-validation accuracy: {cv_score}")
    return

"""
--------------------------------------------------------------------------------------------
WRAPPER FUNCTION
--------------------------------------------------------------------------------------------
"""
def do_everything(user_model, X, y, X_ext, y_ext):
    """This function is created for wrapping purposes and simplify
    the main function. From the model, internal and external datasets,
    returns the fitted model and its scoring. 
    
    Parameters
    ----------
    user_model : str
        User selected model (e.g.: knn) to be parsed to sklearn model
        (KNeighborsClassifier(n_neighbors=2, p=1, weights="distance")

    X : list or pandas dataframe
        Internal dataframe without tags.
    
    y : list or pandas dataframe
        Tags of internal dataframe. 

    X_ext : list or pandas dataframe
        External dataframe without tags.
    
    y_ext : list or pandas dataframe
        Tags of external dataframe.
    
    Returns
    -------
    model : sklearn object
        sklearn model
    
    acc_test : float
        Accuracy of the internal test.

    conf_test : str
        Confusion matrix of the internal test.

    acc_ext : float
        Accuracy of the external test.

    conf_ext : str
        Confusion matrix of the external test.

    cv_score : str
        Mean +/- standard deviation (2 decimals)
    """

    # Retrieve model from user
    model = single_model.retrieve_model(user_model)

    # Method 1: from the 95% remaining 30% test, 70% train.
        # Fits model to train set
        # Predicts with test set
        # Predicts with external data set
    acc_test, conf_test, acc_ext, conf_ext = single_model.train_test_external(model, X, y, X_ext, y_ext)

    # Method 2: 5% external validation, 95% cross-validation (k=5). 
    # Only extracts scoring. None of the built models has seen the
    # external data.
    cv_score = single_model.cross_validation(model, X, y)
        
    return model, acc_test, conf_test, acc_ext, conf_ext, cv_score

"""
--------------------------------------------------------------------------------------------
ARGPARSE
--------------------------------------------------------------------------------------------
"""
def parse_args(parser):
    """
    Parses command line arguments.

    Parameters
    ----------
    parser : ArgumentParser
        Object initialized from argparse.
    """
    # Required arguments
    required_args = parser.add_argument_group('required arguments')

    all_descriptors = ", ".join(DESCRIPTORS)
    des_help = f"Choose from the availiable descriptors using initials: {all_descriptors}; or 'all'."
    required_args.add_argument("-d", "--descriptors", help=des_help, required=True)

    all_models = ", ".join(DESCRIPTORS)
    models_help = f"Choose from the available models using initials: {all_models}; 'all'"
    required_args.add_argument("-m", "--models", help=models_help, required=True)

    # Optional arguments
    parser.add_argument("--csv", help="CSV with Glide features. If flag not provided," 
                                        "assumed to be in current working directory as 'glide_features.csv'.")
    parser.add_argument("-a", "--actives", help="Actives SDF file. If flag not provided," 
                                        "assumed to be in current working directory as 'actives.sdf'.")
    parser.add_argument("-i", "--inactives", help="Inactives SDF file. If flag not provided," 
                                        "assumed to be in current working directory as 'inactives.sdf'.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and score combinations of single classification models by supplying model_input.csv, "
        "active and inactive molecules, descriptors (mordred, fingerprints) and model type (knn, lr, svc, tree, nb). "
    )
    parse_args(parser)
    args = parser.parse_args()
    
    args = parser.parse_args()
    csv = args.csv
    desc = args.descriptors
    mod = args.models
    a = args.actives
    i = args.inactives

    if not csv: csv = GLIDE_FEATURES
    if not a:   a = ACTIVES_SDF
    if not i:   i = INACTIVES_SDF

    score_models(csv, desc, mod, a, i)
