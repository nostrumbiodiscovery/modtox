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

