from functools import partial
from multiprocessing import Pool
import glob, os
import requests
import re
import xml.etree.ElementTree as ET

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
    """Gets the latest file in the CWD with the specified extension.

    Parameters
    ----------
    ext : str
        Such as "*csv".

    Returns
    -------
    str : 
        If there is a file with specified extension.
    
    None : 
        If not.

    """
    list_of_files = glob.glob(os.path.join(os.getcwd(), ext)) 
    try:
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except ValueError:
        return None

def smiles2inchi(smiles):
    """Accesses the chemspider API to convert the smiles to 
    a InChI.

    Parameters
    ----------
    smiles : str
        Molecule SMILES.

    Returns
    -------
    str
        Molecule InChI.
    """
    url = "https://www.chemspider.com/InChI.asmx/SMILESToInChI"
    xml = requests.get(url, params={"smiles": smiles}).text
    return ET.fromstring(xml).text


