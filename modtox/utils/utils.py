from functools import partial
from multiprocessing import Pool
import glob, os

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


def get_latest_file(ext):
    list_of_files = glob.glob(os.path.join(os.getcwd(), ext)) 
    try:
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except ValueError:
        return None
