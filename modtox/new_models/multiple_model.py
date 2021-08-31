import argparse
import pandas as pd
import os

from modtox.modtox.new_models.glide_processing import process_glide
from modtox.modtox.new_models import features
from modtox.modtox.new_models import single_model
from modtox.modtox.utils import utils

MODELS = ["knn", "lr", "svc", "tree", "nb"]
DESCRIPTORS = ["glide", "mordred", "fingerprints"]
CWD = os.getcwd()

ACTIVES_SDF = "actives.sdf"
INACTIVES_SDF = "inactives.sdf"
EXT_PROP = 0.05

def score_models(glide_features, descriptors_inp, models_inp, active_sdf=ACTIVES_SDF, inactive_sdf=INACTIVES_SDF, ext_prop=EXT_PROP, savedir=CWD):
    process_glide(glide_features, active_sdf, inactive_sdf, savedir=savedir)
    balanced_glide_csv = os.path.join(savedir, "balanced_glide.csv")

    sel_descriptors = retrieve_descriptors(descriptors_inp)
    sel_models = retrieve_models(models_inp)
    
    features_df_dict = features.all_combs(sel_descriptors, balanced_glide_csv, active_sdf, inactive_sdf, savedir=savedir)

    cols = [
        "Glide_features",
        "Mordred_descriptors",
        "Topological_fingerprints",
        "Model",
        "Accuracy_test",
        "Confusion_matrix_test",
        "Accuracy_external",
        "Confusion_matrix_external"
    ]
    scoring_df = pd.DataFrame(columns=cols)

    for df_name in features_df_dict.keys():
        df = features_df_dict[df_name] 
        print(f"For {df_name}...")
        main_df, external_df, train, int, ext = single_model.preprocess(df, ext_prop=ext_prop)
        for model in sel_models:
            model = single_model.fit_model(model, train)
            y_pred_int = model.predict(int[0])
            y_pred_ext = model.predict(ext[0])
            model_str = " ".join(str(model).split()) 
            print(f"\tFor {model_str}:")

            acc_int, conf_int = single_model.score_set(int[1], y_pred_int)
            print(f"\t\tInternal test accuracy: {acc_int}")
            print(f"\t\tInternal test confusion matrix: {conf_int}")

            acc_ext, conf_ext = single_model.score_set(ext[1], y_pred_ext)
            print(f"\t\tExternal test accuracy: {acc_ext}")
            print(f"\t\tExternal test confusion matrix: {conf_ext}")

            d = {
                "Glide_features": "glide" in df_name,
                "Mordred_descriptors": "mordred" in df_name,
                "Topological_fingerprints": "fingerprints" in df_name,
                "Model": model_str,
                "Accuracy_test": acc_int,
                "Confusion_matrix_test": conf_int,
                "Accuracy_external": acc_ext,
                "Confusion_matrix_external": conf_ext
            }

            scoring_df = scoring_df.append(d, ignore_index=True)

    scoring_df["Accuracy_test"] = pd.to_numeric(scoring_df["Accuracy_test"], errors="coerce", downcast="float")        
    scoring_df = scoring_df.sort_values("Accuracy_test", ascending=False)
    utils.save_to_csv(scoring_df, "Descriptors/models score", os.path.join(savedir, "scoring.csv"))
    return scoring_df


def retrieve_models(models_input):
    # Dictionary -> initial: model_name
    d = {model[0]: model for model in MODELS}
    if models_input == "all":
        selected_models = MODELS
    else:
        selected_models = [ d[init] for init in d.keys() if init in models_input ]
    
    return selected_models

def retrieve_descriptors(descriptors_input):
    d = {desc[0]: desc for desc in DESCRIPTORS}
    if descriptors_input == "all":
        selected_descriptors = DESCRIPTORS
    elif descriptors_input == "none":
        selected_descriptors = None
    else:
        selected_descriptors = [ d[init] for init in d.keys() if init in descriptors_input ]

    return selected_descriptors


def parse_args(parser):
    """
    Parses command line arguments.

    Parameters
    ----------
    parser : ArgumentParser
        Object initialized from argparse.
    """
    parser.add_argument("--csv", help="CSV with Glide features.")
    
    all_descriptors = ", ".join(DESCRIPTORS)
    des_help = f"Choose from the availiable descriptors using initials: {all_descriptors}; 'none' or 'all'."
    parser.add_argument("-d", "--descriptors", help=des_help)

    all_models = ", ".join(DESCRIPTORS)
    models_help = f"Choose from the available models using initials: {all_models}; 'all'"
    parser.add_argument("-m", "--models", help=models_help)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and score combinations of single classification models by supplying model_input.csv, "
        "descriptors (mordred, fingerprints) and and model type (knn, lr, svc, tree, nb). "
    )
    parse_args(parser)
    args = parser.parse_args()
    run(args.csv, args.descriptors, args.models)
