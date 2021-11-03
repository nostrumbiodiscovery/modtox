from modtox.modtox.utils.enums import Database, StandardTypes
import pandas as pd
import urllib.parse
import urllib.request


from modtox.modtox.utils._custom_errors import ServerError, BadRequestError, UnsupportedStandardType
from modtox.modtox.Molecules.act import Standard, Activity
from modtox.modtox.Retrievers.retrieverABC import Retriever
from modtox.modtox.utils import utils as u

class RetrieveChEMBL(Retriever):
    def __init__(self) -> None:
        self.ids = dict()
        self.activities = list()

    def retrieve_by_target(self, target):
        """Wrapper function for the retrieval by target. 

        Parameters
        ----------
        target : str
            UniProt accession code. 

        """
        try: 
            chemblid = self._uniprot2chembl(target)
            unparsed_activities = self._request_by_target(chemblid)
            
            for unparsed_activity in unparsed_activities:
                activity = self._parse_activity(unparsed_activity)
                self.activities.append(activity)
            
        except ServerError as e:
            print(e)
        except BadRequestError as e:
            print(e)


    
    def _uniprot2chembl(self, target):
        """Converts the UniProt A/C to the ChEMBL target ID.
        See UniProt mapping API for details: https://www.uniprot.org/help/api_idmapping
        Parameters
        ----------
        target : str
            UniProt A/C

        Returns
        -------
        str
            ChEMBL target ID

        Raises
        ------
        BadRequestError
            If the target is not present in ChEMBL database.
        """
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
        """Requests to the ChEMBL Python API.

        Parameters
        ----------
        chemblid : str
            ChEMBL target ID

        Returns
        -------
        List[Dict[str, any]]
            List of dictionary (records) for each activity in the ChEBML format.

        Raises
        ------
        ServerError
            If it is impossible to access the API.
        """
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
        """Converts a activity to a modtox.Activity object. Each activity contains:
            - modtox.Standard
            - molecule (InChI)
            - database
            - target
        Also, adds the ChEMBL ID to the self.ids dict.

        Parameters
        ----------
        unparsed_activity : Dict
            Activity in the ChEMBL format (row).
            
        Returns
        -------
        modtox.Activity
            If the normalization of the standard is successful.
        """
        if (unparsed_activity["standard_value"] is not None and    # Conditions for parsing activity
            unparsed_activity["standard_value"] != 0):

            smile = unparsed_activity["canonical_smiles"]
            if smile is None:
                return
            inchi = u.smiles2inchi(smile)
            self.ids[inchi] = unparsed_activity["molecule_chembl_id"]
            std = self._normalize_standard(unparsed_activity)

        return Activity(
            inchi=inchi, 
            standard=std, 
            database=Database.ChEMBL, 
            target=self.target)
    
    @staticmethod
    def _normalize_standard(unparsed_activity) -> Standard:
        """Converts activity from ChEMBL format to modtox.Standard.

        Parameters
        ----------
        unparsed_activity : Dict
            Activity in the PubChem format (row).

        Returns
        -------
        modtox.Standard
            
        Raises
        ------
        UnsupportedStandardType
            If standard not in the StandardTypes enum class.

        """
        std_type = unparsed_activity["standard_type"].replace(" ", "").replace("/", "_")

        if std_type not in StandardTypes._member_names_:
            raise UnsupportedStandardType(
                f"Standard type '{std_type}' not supported."
            )
        std_type = StandardTypes[std_type]
        std_rel = unparsed_activity["standard_relation"]
        std_val = float(unparsed_activity["standard_value"])
        std_unit = unparsed_activity["standard_units"]
        return Standard(std_type=std_type, std_rel=std_rel, std_val=std_val, std_unit=std_unit)

        # Maybe it should be inside a try: block, excepting any exception and returning 
        # None if something fails, same as PubChem retriever. 

    

