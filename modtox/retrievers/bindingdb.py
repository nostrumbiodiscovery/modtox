import requests
import json

from modtox.modtox.utils.enums import Database, StandardTypes
from modtox.modtox.utils._custom_errors import ServerError, BadRequestError, UnsupportedStandardRelation
from modtox.modtox.Molecules.act import Standard, Activity
from modtox.modtox.constants import constants as k
from modtox.modtox.Retrievers.retrieverABC import Retriever
from modtox.modtox.utils import utils as u

class RetrieveBDB(Retriever):
    def __init__(self) -> None:
        super().__init__()
        self.ids = dict()
        self.activities = list()

    def retrieve_by_target(self, target):
        """Wrapper function for the retrieval by target. 

        Parameters
        ----------
        target : str
            UniProt A/C
        """
        try:
            req = self._request_by_target(target)
            self._get_unparsed_activities(req)
            for unparsed_activity in self.unparsed_activities:
                activity = self._parse_activity(unparsed_activity)
                if activity is None:  # If activity is unparseable
                    continue
                self.activities.append(activity)  # If activity is parseable

        except ServerError as e:
            print(e)
        except BadRequestError as e:
            print(e)

    def _request_by_target(self, target):
        """Accesses the BindingDB API (web services)

        Parameters
        ----------
        target : str
            UniProt A/C

        Returns
        -------
        request object
            Result of the request.

        Raises
        ------
        BadRequestError
            If target not in database.
        ServerError
            If server not accessible. It usually happens.
        """
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
        """Gets the request's json and reads it.

        Parameters
        ----------
        req : request object
            Request object resulting from the database request.
        """
        parsed_json = json.loads(req.text)
        self.unparsed_activities = parsed_json['getLigandsByUniprotsResponse']['affinities']  # List[Dict[str,any]]
    
    def _parse_activity(self, unparsed_activity):
        """Transforms 2-element standards to 4-element standards and creates
        Activity instances. e.g: (IC50 123.1) -> (IC50 = 123.1 nM)

        Parameters
        ----------
        unparsed_activity : Dict[str, any]
            Activity in BindingDB format.

        Returns
        -------
        modtox.Activity

        """
        
        # Get activity information
        bdbm = unparsed_activity["monomerid"]
        smiles = unparsed_activity["smile"]
        if smiles is None:  
            return None  # If activity does not have a SMILES associated.

        inchi = u.smiles2inchi(smiles)  # Convert the SMILES to InChi using chemspider API.
        query = unparsed_activity["query"]  # target

        # Get affinity and create standard
        std_type = unparsed_activity["affinity_type"]
        std_val = unparsed_activity["affinity"]
        std_type, std_rel, std_val, std_unit = self._normalize_standard(std_type, std_val)
        std = Standard(std_type=std_type, std_rel=std_rel, std_val=std_val, std_unit=std_unit)
        if std is None:
            return None  # If standard is unparseable.

        self.ids[inchi] = f"BDBM{bdbm}"  # Add ID to self.ids

        return Activity(
            inchi=inchi, 
            standard=std, 
            database=Database.BindingDB, 
            target=query)
        
    @staticmethod
    def _normalize_standard(std_type, std_val):
        """Converts the activity retrieved from the database to a normalized
        modtox.Standard. From database is a two-element standard (std_type and
        std_val). Considerations:
            - If std_rel is not "=", it is the first element of the std_val.
            - Units are different for each standard type. Mapping in constants.

        Parameters
        ----------
        std_type : str
            
        std_val : str
            Can contain the std_rel.

        Returns
        -------
        modtox.Standard
            If all conversions are possible, that is, parseable activity.
        
        None
            If something fails.
        """
        try:
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
        except Exception:
            return None



    
    
