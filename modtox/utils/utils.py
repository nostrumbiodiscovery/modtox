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

def save_to_csv(dataframe, df_name, filename):
    """
    Saves a dataframe to csv and prints it to terminal. 
    """
    file_name = str(filename)
    dataframe.to_csv(file_name)
    print(f"{str(df_name).capitalize()} saved to {file_name}!")

def merge_ifempty(df1, df2, where):
    '''
    Merges dataframes, checking if are empty. 
    '''
    try:
        # Try to merge both
        result = pd.merge(df1, df2, on=where)
    except:
        # If both empty, return empty DataFrame
        if df1.empty and df2.empty:
            result = pd.DataFrame()
        # Else if, return the only populated DataFrame
        elif df1.empty:    
            result = df2
        elif df2.empty:
            result = df1
    return result

def drop_before_column(dataframe, column):

    dataframe.columns = dataframe.columns.str.replace(" ","_")

    # Drops all columns before specified column
    cols = dataframe.columns.to_list()
    i = cols.index(column)

    # Converts to float before dropping
    for col in cols[i+1:]:
        dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce", downcast="float")
    
    dataframe.drop(columns=cols[0:i], inplace=True)
    # Resets index and fills NaN
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe
    
