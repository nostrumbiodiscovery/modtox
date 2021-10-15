from modtox.modtox.new_models_classes.Features.add_feat import (
    AddFeature,
    AddGlide,
    AddMordred,
    AddTopologicalFingerprints,
)
from modtox.modtox.new_models_classes.mol import BaseMolecule, MoleculeFromChem
from modtox.modtox.new_models_classes.Features.feat_enum import Features
from modtox.modtox.new_models_classes._custom_errors import FeatureError

from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
import pandas as pd
import random
from typing import List
import os
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


# NO UNIT TESTS FOR COLLECTION
class BaseCollection:
    "Object gathering the features of collection of molecules."

    glide_features_csv: str
    molecules: List[BaseMolecule]

    def __init__(self) -> None:
        pass

    def add_features(self, *features: Features, glide_csv=None):
        if not features:
            raise FeatureError("Must specify which features to calculate.")

        feat_map = {  # Maybe move this to feat_enum for consistency.
            Features.glide: AddGlide,
            Features.mordred: AddMordred,
            Features.topo: AddTopologicalFingerprints,
        }

        for ft in features:
            feat_dict = feat_map[ft](
                self.molecules, glide_csv=glide_csv
            ).calculate()  # glide_csv only needed for glide features, but others accept kwargs.
            for mol in self.molecules:
                mol.add_feature(ft, feat_dict[mol])

    def cluster(self, cutoff=0.4, fp: str = "topological"):
        if fp == "topological" and not hasattr(self.molecules[0], "topo"):
            fps = [Chem.RDKFingerprint(x.molecule) for x in self.molecules]
        elif fp == "topological" and hasattr(self.molecules[0], "topo"):
            fps = [mol.topo for mol in self.molecules]
        elif fp == "morgan":
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol.molecule, 2, 1024) for mol in self.molecules]
        else:
            raise NotImplementedError("Implemented fingerprints for clustering are: 'topological' and 'morgan'.")

        dists = []
        nfps = len(fps)
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
            dists.extend([1-x for x in sims])

        self.clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
        
        for id, cluster in enumerate(self.clusters):
            members = len(cluster)
            for i, mol_idx in enumerate(cluster):
                is_centroid = i == 0
                self.molecules[mol_idx].add_cluster_info(id, members, is_centroid)
    
    def to_dataframe(self, *features):
            """Converts collection to df to build the model.
            If no features are supplied, all features are added
            (see Molecule.to_record() method)."""
            
            if not hasattr(self, "clusters"):
                self.cluster()

            records = list()
            for molecule in self.molecules:
                records.append(molecule.to_record(*features))
            df = pd.DataFrame(records)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast="float", errors="coerce")
            return df


class CollectionFromSDF(BaseCollection):
    """Builds the collection from two SDF files."""

    def __init__(self, actives_sdf, inactives_sdf) -> None:
        self.read_sdf(actives_sdf, inactives_sdf)
        self.molecules = self.actives + self.inactives

        self.features_added = list()
        self.dataframes = list()

    def read_sdf(self, actives_sdf, inactives_sdf):
        self.actives = [
            MoleculeFromChem(mol)
            for mol in Chem.SDMolSupplier(actives_sdf)
            if ":" in mol.GetProp("_Name")
        ]
        self.inactives = [
            MoleculeFromChem(mol)
            for mol in Chem.SDMolSupplier(inactives_sdf)
            if ":" in mol.GetProp("_Name")
        ]


class CollectionFromTarget(BaseCollection):
    """Builds a collection by target retrieving
    by target UniProt accession code."""
    pass


class CollectionSummarizer:
    """Class responsible for formatting output for user review."""

    def __init__(self, collection: BaseCollection, savedir=os.getcwd()) -> None:
        self.collection = collection
        self.savedir = savedir

    def plot_similarities(self, bins=15):
        actives_sim = [
            mol.similarity for mol in self.collection.molecules if mol.activity == True
        ]
        inactives_sim = [
            mol.similarity for mol in self.collection.molecules if mol.activity == False
        ]
        x = np.array([actives_sim, inactives_sim])
        plt.hist(x, bins=bins, stacked=True, density=True)
        plt.title("Collection similarity")
        plt.legend(["Actives", "Inactives"])
        plt.show()
        plt.savefig(os.path.join(self.savedir, "histogram_similarity.png"))

    def draw_most_representative_scaffolds(self, n):
        scaffolds = [
            mol.scaffold for mol in self.collection.molecules if mol.scaffold != ""
        ]
        d = Counter(scaffolds).most_common(n)
        mols = [Chem.MolFromSmiles(smiles[0]) for smiles in d]
        leg = [f"Count: {count[1]}" for count in d]
        img = MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), legends=leg)
        img.save(os.path.join(self.savedir, "representative_scaffolds.png"))
        return

    def plot_scaffolds(self):
        scaffolds = {
            mol: mol.scaffold for mol in self.collection.molecules if mol.scaffold != ""
        }
        Counter(scaffolds)

    def draw_cluster(self):
        to_draw = [self.molecules[idx].molecule for idx in self.clusters[0]]
        for mol in to_draw:
            AllChem.Compute2DCoords(mol)
        img = MolsToGridImage(to_draw)
        img.save('test.png')