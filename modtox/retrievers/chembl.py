from modtox.modtox.utils.enums import Database, StandardTypes
import pandas as pd
import urllib.parse
import urllib.request


from modtox.modtox.utils._custom_errors import ServerError, BadRequestError, UnsupportedStandardType
from modtox.modtox.Molecules.act import Standard, Activity
from modtox.modtox.Retrievers.retrieverABC import Retriever, RetSum
from modtox.modtox.utils import utils as u

class RetrieveChEMBL(Retriever):
    def __init__(self) -> None:
        self.ids = dict()
        self.activities = list()

    def retrieve_by_target(self, target):
        """Wrapper function, returns query summary."""
        try: 
            chemblid = self._uniprot2chembl(target)
            unparsed_activities = self._request_by_target(chemblid)
            request = "Successful"
            
            for unparsed_activity in unparsed_activities:
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
    
    def _uniprot2chembl(self, target):
        self.target = target
        url = 'https://www.uniprot.org/uploadlists/'
        params = {
        'from': 'ACC+ID',
        'to': 'CHEMBL_ID',
        'format': 'tab',
        'query': target
        }

        data = urllib.parse.urlencode(params)
        data = data.encode('utf-8')
        
        req = urllib.request.Request(url, data)
        with urllib.request.urlopen(req) as f:
            response = f.read()

        response = response.decode('utf-8')
        if response == "From\tTo\n":
            raise BadRequestError(db=Database.ChEMBL.name, uniprot=target)

        chemblid = response.split()[3]
        return chemblid

    def _request_by_target(self, chemblid):
        try:
            from chembl_webresource_client.new_client import new_client
            activity = new_client.activity
            cols = [
                "canonical_smiles",
                "molecule_chembl_id",
                "standard_type",
                "standard_relation", 
                "standard_units", 
                "standard_value"
            ]
            query = activity.filter(target_chembl_id=chemblid)
            df = pd.DataFrame.from_dict(query)
            df = df[df["standard_value"].notna()]
            df = df[cols]
            unparsed_activities = df.to_dict("records")
            return unparsed_activities
        except ConnectionError:
            raise ServerError(Database.ChEMBL.name)
        
    def _parse_activity(self, unparsed_activity):
        if (unparsed_activity["standard_value"] is not None and    # Conditions for appending
            unparsed_activity["standard_value"] != 0):

            smile = unparsed_activity["canonical_smiles"]
            inchi = u.smiles2inchi(smile)
            self.ids[inchi] = unparsed_activity["molecule_chembl_id"]
            std = self._normalize_standard(unparsed_activity)

        return Activity(
            inchi=inchi, 
            standard=std, 
            database=Database.BindingDB, 
            target=self.target)
    
    @staticmethod
    def _normalize_standard(unparsed_activity) -> Standard:
        std_type = unparsed_activity["standard_type"]
        
        if std_type not in StandardTypes._member_names_:
            raise UnsupportedStandardType(
                f"Standard type '{std_type}' not supported."
            )
        std_type = StandardTypes[unparsed_activity["standard_type"]]
        std_rel = unparsed_activity["standard_relation"]
        std_val = float(unparsed_activity["standard_value"])
        std_unit = unparsed_activity["standard_units"]
        return Standard(std_type=std_type, std_rel=std_rel, std_val=std_val, std_unit=std_unit)

    

