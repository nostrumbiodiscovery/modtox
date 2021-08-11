from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

COLUMNS_TO_EXCLUDE = ["Title", "Rank", "ID", "Unnamed: 0"]


def parallelize(func, iterable, n_workers, **kwargs):
    """
    Parallelize execution of a function.

    Parameters
    ----------
    func : function
        Function to parallelize.
    iterable : List[Any]
        List of function inputs.
    n_workers : int
        Number of CPUs to use.

    Returns
    -------
        Output of the function.
    """
    f = partial(func, **kwargs)

    with Pool(n_workers) as p:
        output = p.map(f, iterable=iterable)

    return output


def get_data(csv):
    """
    Reads in the model data from features CSV, drops irrelevant columns, converts everything to numerical values,
    encodes labels, and saves X and y as numpy binary files.

    Parameters
    ----------
    csv : str
        Path to CSV file containing model input.

    Returns
    -------
        Arrays with X and y.
    """

    # Read in the model input and separate out the labels
    activity_column = "Activity"
    df = pd.read_csv(csv, dtype=object)

    print("Dropping and converting columns...")
    # Drop irrelevant columns
    numbered_columns = [column for column in df.columns if "#" in column]

    for column in numbered_columns + COLUMNS_TO_EXCLUDE:
        df.drop(column, axis=1, inplace=True)

    # Assign labels column to y and drop it from the main dataframe
    y = df[activity_column]
    df.drop(activity_column, axis=1, inplace=True)

    # Convert labels from True/False to 0/1
    le = LabelEncoder()
    y_trans = le.fit_transform(y)

    # Convert to numeric, replace non-convertible strings with NaM
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce", downcast="float")

    X = df
    X.to_csv("X.csv")

    print(f"Finished processing the input data. X {X.shape}, y {y_trans.shape}")
    np.save("X.npy", X)
    np.save("y.npy", y_trans)

    return X, y_trans
