import requests
import json

from modtox.modtox.utils.enums import Database, StandardTypes
from modtox.modtox.utils._custom_errors import ServerError, BadRequestError, UnsupportedStandardRelation
from modtox.modtox.Molecules.act import Standard, Activity
from modtox.modtox.constants import constants as k
from modtox.modtox.Retrievers.retrieverABC import Retriever, RetSum
from modtox.modtox.utils import utils as u

class RetrieveBDB(Retriever):
    def __init__(self) -> None:
        super().__init__()
        self.ids = dict()
        self.activities = list()

    def retrieve_by_target(self, target):
        """Wrapper function, returns query summary."""
        try:
            req = self._request_by_target(target)
            request = "Successful"
            self._get_unparsed_activities(req)
            for unparsed_activity in self.unparsed_activities:
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
        
    def _request_by_target(self, target):
        """Requests from BindingDB API"""
        url = (
            "http://www.bindingdb.org/axis2/services/BDBService/getLigandsByUniprots"
            f"?uniprot={target}&cutoff=10000000&code=0&response=application/json"
        )
        req = requests.get(url)
        if req.status_code == 500:
            raise BadRequestError(Database.BindingDB, target)
        elif req.status_code != 200 and req.status_code != 500:
            raise ServerError(Database.BindingDB.name)
        return req

    def _get_unparsed_activities(self, req):
        parsed_json = json.loads(req.text)
        self.unparsed_activities = parsed_json['getLigandsByUniprotsResponse']['affinities']
    
    def _parse_activity(self, unparsed_activity):
        """Transforms 2-element standards to 4-element standards and creates
        Activity instances
        (IC50 123.1) -> (IC50 = 123.1 nM) """
        
        # Get activity information
        bdbm = unparsed_activity["monomerid"]
        smiles = unparsed_activity["smile"]
        inchi = u.smiles2inchi(smiles)
        query = unparsed_activity["query"]

        # Get affinity and create standard
        std_type = unparsed_activity["affinity_type"]
        std_val = unparsed_activity["affinity"]
        std_type, std_rel, std_val, std_unit = self._normalize_standard(std_type, std_val)
        std = Standard(std_type=std_type, std_rel=std_rel, std_val=std_val, std_unit=std_unit)
        
        self.ids[inchi] = f"BDBM{bdbm}"

        return Activity(
            inchi=inchi, 
            standard=std, 
            database=Database.BindingDB, 
            target=query)
        
    @staticmethod
    def _normalize_standard(std_type, std_val):
        if std_type not in k.bdb_units:
            raise UnsupportedStandardRelation(
                f"Standard relation '{std_val[0]}' not supported."
            )
        
        std_val = str(std_val).strip()
        if std_val[0].isdigit():
            std_val = float(std_val)
            std_rel = "="
        elif std_val.startswith(">"):
            std_rel = ">"
            std_val = str(std_val)[1:]
            std_val = float(std_val)
        elif std_val.startswith("<"):
            std_rel = "<"
            std_val = str(std_val)[1:]
            std_val = float(std_val)

        std_unit = k.bdb_units[std_type]
        std_type = StandardTypes[std_type]
    
        return std_type, std_rel, std_val, std_unit


    
    
