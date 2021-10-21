
from selenium import webdriver
import time
from tqdm import tqdm
import os
import pubchempy as pcp
import json

from modtox.modtox.utils import utils as u
from modtox.modtox.utils.enums import Database, StandardTypes
from modtox.modtox.utils._custom_errors import ServerError, BadRequestError, UnsupportedStandardType
from modtox.modtox.Molecules.act import Standard, Activity
from modtox.modtox.Retrievers.retrieverABC import Retriever, RetSum

dirdirname = os.path.dirname(os.path.dirname(__file__))
DRIVER_PATH = os.path.join(dirdirname, "utils/chromedrivers/macOS_chromedriver")

class RetrievePubChem(Retriever):
    def __init__(self) -> None:
        super().__init__()
        self.ids = dict()
        self.activities = list()
        self.tagged_activities = dict()
    def retrieve_by_target(self, target):
        try:
            request = self._download_json(target)
            unparsed_activities = self._read_json()
            for unparsed_activity in tqdm(unparsed_activities):
                if "acname" in unparsed_activity.keys():
                    activity = self._parse_activity(unparsed_activity)
                    self.activities.append(activity)

        except ServerError as e:
            print(e)
            request = "Server Error"
        except BadRequestError as e:
            print(e)
            request = "No hits found"

        return RetSum(
            request=request, 
            retrieved_molecules=len(self.ids), 
            retrieved_activities=len(self.activities))

    def _download_json(self, target):
        self.target = target
        options = webdriver.ChromeOptions()
        prefs = {
            "download.default_directory" : os.getcwd(),

        }
        options.add_experimental_option("prefs",prefs)
        # options.add_argument("--headless")
        
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
            return "Successful"

        except Exception:
            raise BadRequestError(Database.PubChem.name, self.target)
        finally:
            driver.close() 
        
    def _parse_activity(self, unparsed_activity):
        if unparsed_activity["acname"] != 0:
            
            cid = unparsed_activity["cid"]
            inchi = self._cid2inchi(cid)
            std = self._normalize_standard(unparsed_activity)
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
            database=Database.BindingDB, 
            target=target)
    
    def _read_json(self):
        file = u.get_latest_file("*json")
        with open(file, "r") as f:
            unparsed_activities = json.load(f)        
        os.remove(file)
        return unparsed_activities

    @staticmethod   
    def _cid2inchi(cid):
        c = pcp.Compound.from_cid(cid)
        return c.inchi
    
    @staticmethod
    def _normalize_standard(unparsed_activity):
        std_type = unparsed_activity["acname"]
        
        if std_type not in StandardTypes._member_names_:
            raise UnsupportedStandardType(
                f"Standard type '{std_type}' not supported."
            )

        std_type = StandardTypes[std_type]
        std_rel = unparsed_activity["acqualifier"]
        std_val = float(unparsed_activity["acvalue"]) * 1000 # µM to nM
        std_unit = "nM"
        return Standard(std_type=std_type, std_rel=std_rel, std_val=std_val, std_unit=std_unit)