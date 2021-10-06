from modtox.modtox.new_models import single_model
from modtox.modtox.new_models._parameters import *

from sklearn.model_selection import cross_val_score
import pandas as pd


def generate_single_model(csv, user_model, ext_prop=EXTERNAL_PROPORTION):
    """
    Fits the supplied data into the model specified by the user. 
    
    Parameters 
    ----------
    csv : str 
        Path to CSV file with any combination of data from three dataframes: 
        Glide features, mordred descriptors and topological fingerprints.
    
    user_model : str
        User selected model between knn, lr, svc, tree or nb.

    ext_prop : float, default = 0.05
        Proportion of external set extracted to validate the model. 
        Default is set at '_parameters.py' file.

    Returns
    -------

    """
    print(f"Pre-processing {csv} (dropping columns, imputation and splitting into sets)... ", end="")
    df = pd.read_csv(csv, index_col=0)

    main_df, external_df, X_imp, y, X_ext_imp, y_ext = preprocess(df, ext_prop=EXTERNAL_PROPORTION)
    print("Done.")

    model, mean, std = cross_validation(user_model, X_imp, y)
    
    y_pred_ext = model.predict(X_ext_imp)
    acc_ext, conf_ext = single_model.score_set(y_ext, y_pred_ext)

    print(f"Internal test accuracy: {acc_ext}")
    print(f"Internal test confusion matrix: {conf_ext}")
    
    return main_df, external_df, acc_ext, conf_ext

"""
--------------------------------------------------------------------------------------------
HELPER FUNCTIONS
--------------------------------------------------------------------------------------------
"""
def preprocess(df, ext_prop=EXTERNAL_PROPORTION):
    df = single_model.load_model_input(df)

    main_df, external_df = single_model.extract_external_val(df, ext_prop)
    X, y = single_model.labelXy(main_df)
    X_ext, y_ext = single_model.labelXy(external_df)

    X_imp = single_model.imputation(X)
    X_ext_imp = single_model.imputation(X_ext)

    return main_df, external_df, X_imp, y, X_ext_imp, y_ext

def cross_validation(user_model, X, y):
    model = single_model.retrieve_model(user_model)
    scores = cross_val_score(model, X, y)
    return model, scores.mean(), scores.std()

