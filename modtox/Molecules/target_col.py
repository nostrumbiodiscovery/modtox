from dataclasses import dataclass
from typing import Dict, List
from collections import Counter
import os

from numpy.lib.arraysetops import isin

from modtox.modtox.Molecules.mol import MoleculeFromInChI
from modtox.modtox.Retrievers.bindingdb import RetrieveBDB
from modtox.modtox.Retrievers.chembl import RetrieveChEMBL
from modtox.modtox.Retrievers.pubchem import RetrievePubChem
from modtox.modtox.utils.enums import Database, Features, StandardTypes
from modtox.modtox.constants import constants as k
from modtox.modtox.Molecules.col import BaseCollection

from matplotlib_venn import venn3_unweighted
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
import statistics
import pickle5


class IDs:
    """Dataclass for storing and merging IDs. 
    """
    def __init__(self) -> None:
        self.dbs = list()
    
    def __str__(self):
        s = ""
        for db in self.dbs:
            id = getattr(self, f"{(db.name).lower()}_id")
            s += f"\n{db.name}: {id}"
        return s

class CollectionFromTarget(BaseCollection):
    """Collection created by searching in databases (with UniProt AC)
    """
    molecules: List[MoleculeFromInChI]

    def __init__(self, target: str) -> None:
        """[summary]

        Parameters
        ----------
        target : str
            UniProt accession code.
        """
        self.target = target
        self.molecules = list()
        self.activity_criteria = dict()

    def fetch(self, dbs: List[Database]):
        """Fetches from the selected databases. Gathers all activities in a list
        (each activity has one InChI associated) and ids in a dict {db -> {inchi -> id}}

        Parameters
        ----------
        dbs : List[Database] (enum class)
            List of the databases to search. 
        """
        db_map = {
            Database.BindingDB: RetrieveBDB,
            Database.ChEMBL: RetrieveChEMBL,
            Database.PubChem: RetrievePubChem,
        }
        self.retrievers = {db: db_map.get(db)() for db in dbs}

        for ret in self.retrievers.values():
            ret.retrieve_by_target(self.target)

        self.activities = [
            activity
            for retriever in self.retrievers.values()
            for activity in retriever.activities
        ]

        self.ids = {db: ret.ids for db, ret in self.retrievers.items()}

        self.pubchem_activity = self.retrievers[
            Database.PubChem
        ].tagged_activities

    def get_unique_inchis(self):
        """Merges all inchis in a set to avoid repeated values.
        """
        self.unique_inchis = {
            inchi for id_dict in self.ids.values() for inchi in id_dict.keys()
        }

    def _unify_ids(self):
        """Constructs a dict {inchi -> IDs} to map each molecule (inchi) to
        its ids from different databases. 
        """
        self.unified_ids = {inchi: IDs() for inchi in self.unique_inchis}
        for db, ids_dict in self.ids.items():
            for inchi, id in ids_dict.items():
                self.unified_ids[inchi].dbs.append(db)
                setattr(self.unified_ids[inchi], f"{db.name.lower()}_id", id)

    def create_molecules(self):
        """Initializes the modtox.MoleculeFromInChI. Adds all the activities and
        the IDs object to each molecule.
        """
        self.molecules = list()
        for i, inchi in enumerate(self.unique_inchis):
            mol = MoleculeFromInChI(inchi, f"{self.target}_{i}")

            mol.ids = self.unified_ids[inchi]

            for activity in [
                act for act in self.activities if act.inchi == inchi
            ]:
                mol.add_activity(activity)

            self.molecules.append(mol)

    def assess_activity(
        self, criteria: Dict[StandardTypes, float or int]
    ):  # For pubchem, {"pubchem": 0}
        """Assesses the activity for a given number of criterions for each molecule.

        Parameters
        ----------
        criteria : Dict[StandardTypes, float or int]
            Dictionary with all the criterias to evaluate. 
        """
        for std_type, std_val in criteria.items():
            self.activity_criteria[std_type] = std_val
            if std_type == "pubchem":
                self.add_pubchem_activity()
            else:
                for molecule in self.molecules:
                    molecule.assess_activity(std_type, std_val)

    def add_pubchem_activity(self):
        """In case PubChem criterion is selected, directly adds to the molecule the
        activity from the PubChem retrieval result. 
        """
        for molecule in self.molecules:
            act = self.pubchem_activity.get(molecule.inchi)
            if act is None:
                molecule.activity_tags["pubchem"] = "No data"
            elif all(x == "Active" for x in act):
                molecule.activity_tags["pubchem"] = "Active"
            elif all(x == "Inactive" for x in act):
                molecule.activity_tags["pubchem"] = "Inactive"
            else:
                molecule.activity_tags["pubchem"] = "Contradictory data"

    def to_sdf(self, output=os.getcwd()):
        """Saves to SDF using the target as name.

        Parameters
        ----------
        output : str, optional
            Path to output folder, by default CWD.
        """
        if output is None:
            file = "-"  # For printing instead of saving.
        else:
            file = os.path.join(output, f"{self.target}.sdf") 

        w = Chem.SDWriter(file)
        for mol in self.molecules:
            mol._add_properties_to_molecule() 
            w.write(mol.molecule)

    def to_pickle(self, output=os.getcwd()):
        """Saves collection to pickle object to be loaded later.

        Parameters
        ----------
        output : str, optional
            Path to output folder, by default CWD.
        """
        file = os.path.join(output, f"{self.target}.pickle")
        with open(file, "wb") as f:
            pickle5.dump(self, f)

    def summarize_retrieval(self, output=os.getcwd(), save=True):
        """Outputs the plots for assessing retrieval and saves them to
        'retrieval_summary/' folder.

        Parameters
        ----------
        output : str, optional
            Path to output folder, by default CWD
        save : bool, optional
            Save or only show (not tested), by default True
        """
        folder = os.path.join(output, "retrieval_summary")
        if save:
            if not os.path.exists(folder):
                os.makedirs(folder)
            rs = RetrievalSummary(self).save(folder)
        else:
            rs = RetrievalSummary(self)
            rs.show_plots()


class RetrievalSummary:
    """Class responsible for generating and saving the plots.
    """
    def __init__(self, collection: CollectionFromTarget) -> None:
        self.collection = collection

    def show_plots(self):
        """Generates plots and shows them. 
        """
        fig = self._merge_plots()
        plt.show()

    def save(self, outputdir):
        """Generates all plots and tables and saves them.

        Parameters
        ----------
        outputdir : str
            Path to save.
        """
        fig = self._merge_plots()
        fig.savefig(os.path.join(outputdir, "overview.png"))
        acts_table = self._activities_table()
        acts_table.to_csv(
            os.path.join(outputdir, "retrieved_activities.tsv"), sep="\t"
        )
        acts_assessment = self._activity_assessment_table()
        acts_assessment.to_csv(
            os.path.join(outputdir, "activity_assessment.tsv"), sep="\t"
        )

    def _merge_plots(self):
        """Joins venn diagram and pie chart.

        Returns
        -------
        matplotlib.pyplot.figure
            1 by 2 figure of Venn diagram (molecules databases) and std_piechart. 
        """
        fig, axs = plt.subplots(1, 2)
        fig.tight_layout()
        self._molecules_databases_venn(axs[0])
        self._std_types_piechart(axs[1])
        return fig

    def _molecules_databases_venn(self, ax):
        """Generates a unweightened Venn diagram of the databases that the molecules
        were found.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            The axis in which the Venn diagram will be generated.
        """
        inchis = {
            db: set(ids_dict.keys())
            for db, ids_dict in self.collection.ids.items()
        }
        fig = venn3_unweighted(
            [s for s in inchis.values()],
            set_labels=(db.name for db in inchis.keys()),
            ax=ax,
        )
        ax.title.set_text(
            f"Overlay of molecules in databases\nRetrieved molecules: {len(self.collection.unique_inchis)}"
        )

    def _std_types_piechart(self, ax):
        """Creates a piechart with the number of activities
        of each standard type. If there are standard that contribute to
        the 5% or less, are catalogued as others. 

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            The axis in which the Venn diagram will be generated.
        """
        standards = [
            act.standard.std_type for act in self.collection.activities
        ]
        ct = Counter(standards)
        total = sum(ct.values())
        others = sum([val for val in ct.values() if val/total < 0.05])
        relevant = {std: val for std, val in ct.items() if val/total >= 0.05}
        relevant["Others"] = others
        values = list(relevant.values())
        labels = list()
        colors = ["#EDA7AD", "#A3A7F9", "#AFCCAD"]
        for std in relevant.keys():
            if isinstance(std, StandardTypes):
                labels.append(std.name)
            else:
                labels.append(std)

        plot = ax.pie(
            values,
            labels=labels,
            autopct=lambda p: "{:.0f}".format(p * total / 100),  # To show absolute value instead of percentage
            startangle=90,
            colors=colors
        )
        ax.title.set_text(
            f"Retrieved standard types\nRetrieved activities: {total}"
        )

    def _activities_table(self):
        """Summarizes the retrieval by standard types:
            Activities: number of activities retrieved.
            Molecules: number of molecules retrieved.
            Avg stdev: average of the standard deviation of the activities of each molecule.

        Returns
        -------
        pd.DataFrame
            Table with all the information above. 
        """
        acts_per_std = dict()  # {std_type -> List[Activity]}
        
        # Group activities by standard type
        for activity in self.collection.activities:
            if activity.standard.std_type not in acts_per_std:
                acts_per_std[activity.standard.std_type] = [activity]
            else:
                acts_per_std[activity.standard.std_type].append(activity)

        data = dict()
        
        # For each standard type
        for std_type, activities in acts_per_std.items():
            mols = {act.inchi for act in activities}  # Get all the molecules with that std.
            std_devs = list()
            
            for mol in mols:
                acts_per_mol = [  # Get the number of activities for this inchi
                    act.standard.std_val
                    for act in activities
                    if act.inchi == mol
                ]
                if len(acts_per_mol) == 1:
                    std_devs.append(0)  # Stdev is 0 if only 1 activity
                else:
                    std_devs.append(statistics.stdev(acts_per_mol))

            data[std_type.name] = {
                "Activities": len(activities),
                "Molecules": len(mols),
                "Avg stdev": statistics.mean(std_devs),
                "Units": k.bdb_units[std_type.name],
            }

        df = pd.DataFrame(data)
        df = df.transpose()
        df["Activities/molecule"] = df["Activities"] / df["Molecules"]
        return df

    def _activity_assessment_table(self):
        """Summarizes the tagged activity for the criteria supplied.

        Returns
        -------
        pd.DataFrame
            Table with the counts for each category 
            ("Active", "Inactive", "No data", "Contradictory data")
        """
        tags = [mol.activity_tags for mol in self.collection.molecules]
        df = pd.DataFrame(tags)
        counts_df = pd.DataFrame(
            columns=["Active", "Inactive", "No data", "Contradictory data"],
        )
        for col in df.columns:
            counts_df = counts_df.append(df[col].value_counts())
            counts_df.fillna(0, inplace=True)
            counts_df = counts_df.astype(int)
        return counts_df
