
from selenium import webdriver
import time
from tqdm import tqdm
import os
import pubchempy as pcp
import json
from typing import List, Dict

from modtox.modtox.utils import utils as u
from modtox.modtox.utils.enums import Database, StandardTypes
from modtox.modtox.utils._custom_errors import ServerError, BadRequestError
from modtox.modtox.Molecules.act import Standard, Activity
from modtox.modtox.Retrievers.retrieverABC import Retriever

dirdirname = os.path.dirname(os.path.dirname(__file__))
DRIVER_PATH = os.path.join(dirdirname, "utils/chromedrivers/macOS_chromedriver")

class RetrievePubChem(Retriever):
    activities: List[Activity]  # Inherited from abstract base class.
    ids: Dict[str, str]  # Inherited {inchi -> id}

    def __init__(self) -> None:
        super().__init__()
        self.tagged_activities = dict()  
    
    def retrieve_by_target(self, target):
        """Wrapper function for the retrieval by target. 

        Parameters
        ----------
        target : str
            UniProt A/C
        """
        # Download file if possible, print error if raised.
        try:
            self._download_json(target)
            unparsed_activities = self._read_json()
            
            # Parse activities
            for unparsed_activity in tqdm(unparsed_activities):
                if "acname" in unparsed_activity.keys():
                    activity = self._parse_activity(unparsed_activity)
                    if activity is None:
                        continue  # Ommit activity if unparseable. 
                    self.activities.append(activity)  # Append activity if parseable.

        except ServerError as e:
            print(e)
        except BadRequestError as e:
            print(e)

    def _download_json(self, target):
        """Accesses the website using Selenium and downloads the json file.
        Waits to close the driver until download is finished.
        It is important that there is not a file ending in "bioactivity_protein.json"
        in the CWD.

        Parameters
        ----------
        target : str
            UniProt accession code.        

        Raises
        ------
        BadRequestError
            If the database does not have information about that target. 
        """
        self.target = target
        options = webdriver.ChromeOptions()
        prefs = {
            "download.default_directory" : os.getcwd(),

        }
        options.add_experimental_option("prefs",prefs)
        # options.add_argument("--headless")  # To avoid oppening a Chrome window, useless for debug mode.
        
        driver = webdriver.Chrome(DRIVER_PATH, chrome_options=options)
        
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/protein/{target}#section"
            "=Tested-Compounds&fullscreen=true"
        )

        driver.get(url)
        time.sleep(5)
        try:
            driver.find_element_by_xpath('//*[@id="Download"]').click()
            
            driver.find_element_by_xpath('//*[@id="Tested-Compounds"]/div[2]/div[1]'
            '/div/div/div[1]/div[2]/div/div/div[2]/div/div[2]/div/div[2]/a').click()
            
            while not u.get_latest_file("*").endswith("bioactivity_protein.json"):
                time.sleep(1)

        except Exception:
            raise BadRequestError(Database.PubChem.name, self.target)
        finally:
            driver.close() 
        
    def _parse_activity(self, unparsed_activity):
        """Converts a activity to a modtox.Activity object. Each activity contains:
            - modtox.Standard
            - molecule (InChI)
            - database
            - target
        Also, adds the CID to the self.ids dict.

        Parameters
        ----------
        unparsed_activity : Dict
            Activity in the PubChem format (row).
            
        Returns
        -------
        modtox.Activity
            If the normalization of the standard is successful.

        None
            If the normalization is unsucessful.
        """
        if unparsed_activity["acname"] != 0:
            
            cid = unparsed_activity["cid"]
            inchi = self._cid2inchi(cid)
            std = self._normalize_standard(unparsed_activity)
            if std is None:
                return None
            
            target = unparsed_activity["protacxn"]
            
            tagged_activity = unparsed_activity["activityid"]

            self.ids[inchi] = cid
            
            if inchi not in self.tagged_activities.keys():
                self.tagged_activities[inchi] = [tagged_activity]
            else:
                self.tagged_activities[inchi].append(tagged_activity)
            
            return Activity(
            inchi=inchi, 
            standard=std, 
            database=Database.PubChem, 
            target=target)
    
    def _read_json(self):
        """Converts json file to dictionary.

        Returns
        -------
        List[Dict[str, any]]
            List of dictionary (records) for each activity.
        """
        file = u.get_latest_file("*json")
        with open(file, "r", encoding='utf-8') as f:
            unparsed_activities = json.load(f)        
        os.remove(file)
        return unparsed_activities

    @staticmethod   
    def _cid2inchi(cid):
        """The json file does not include the InChI. For that, the PubChem
        API is accessed by compound ID and the InChI is then retrieved.

        Parameters
        ----------
        cid : str
            PubChem compound ID.

        Returns
        -------
        str
            InChI
        """
        c = pcp.Compound.from_cid(cid)
        return c.inchi
    
    @staticmethod
    def _normalize_standard(unparsed_activity):
        """Converts the activity retrieved from the database to a normalized
        modtox.Standard.

        Parameters
        ----------
        unparsed_activity : Dict
            Activity in the PubChem format (row).

        Returns
        -------
        modtox.Standard
            If all conversions are possible.
        
        None
            If something fails. To consider:
            - More std_types can be added to the enum class. 
            - Replace method to convert 1 / K obs -> 1_Kobs (ugly fix).
            - There can not be spaces in the instances of enum classes. 
            - Some unparsed activities are missing some fields.
            - Units are µM, so * 1000.

        """
        try: 
            std_type = unparsed_activity["acname"].replace(" ", "").replace("/", "_")
            std_type = StandardTypes[std_type]
            std_rel = unparsed_activity["acqualifier"]
            std_val = float(unparsed_activity["acvalue"]) * 1000 # µM to nM
            std_unit = "nM"
            return Standard(std_type=std_type, std_rel=std_rel, std_val=std_val, std_unit=std_unit)
        except Exception:
            return None
        
