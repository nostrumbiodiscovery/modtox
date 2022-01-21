from modtox.modtox.Molecules.add_feat import (
    AddFeature,
    AddGlide,
    AddMordred,
    AddMorganFingerprints,
    AddTopologicalFingerprints,
)
from modtox.modtox.Molecules.mol import BaseMolecule, MoleculeFromChem
from modtox.modtox.utils.enums import Features, Database
from modtox.modtox.utils._custom_errors import FeatureError


from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
import pandas as pd
from typing import List
import os
import matplotlib.pyplot as plt
import numpy as np

from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


class BaseCollection:
    "Object gathering the features of collection of molecules."

    glide_features_csv: str
    molecules: List[BaseMolecule]

    def __init__(self) -> None:
        pass

    def add_features(self, *features: Features, glide_csv=None):
        """Calculates the features specified and adds them to the molecule 
        object.

        Parameters
        ----------
        glide_csv : str, optional
            Path to 'glide_features.csv' in case glide features are added, by 
            default None

        Raises
        ------
        FeatureError
            If user does not supply features to calculate.
        """
        if not features:
            raise FeatureError("Must specify which features to calculate.")

        feat_map = {  # Maybe move this to feat_enum for consistency.
            Features.glide: AddGlide,
            Features.mordred: AddMordred,
            Features.topo: AddTopologicalFingerprints,
            Features.morgan: AddMorganFingerprints,
        }

        for ft in features:
            feat_dict = feat_map[ft](
                self.molecules, glide_csv=glide_csv
            ).calculate()  # glide_csv only needed for glide features, but others accept kwargs. 
            for mol in self.molecules:
                mol.add_feature(ft, feat_dict[mol])

    def cluster(self, cutoff=0.4, fp: str = "topological"):
        """Clusters the molecules of the collection by the specified fingerprints.
        Adds the cluster information to each molecule of the collection.

        Parameters
        ----------
        cutoff : float, optional
            Distance between clusters, by default 0.4
        fp : str, optional
            Type of fingerprints, by default "topological"

        Returns
        -------
        clusters
            Result of the Butina clustering. Tuples contains the indexes of the 
            molecules within each cluster. First index of each cluster corresponds
            to the clustering centroid.

        Raises
        ------
        NotImplementedError
            If fp is not topological or morgan.
        """
        if fp not in ["topological", "morgan"]:
            raise NotImplementedError(
                "Implemented fingerprints for clustering are: 'topological' and 'morgan'."
            )

        if fp == "topological":
            if not hasattr(self.molecules[0], "topo"):
                self.add_features(Features.topo)

            fps = [mol.topo for mol in self.molecules]

        elif fp == "morgan":
            if not hasattr(self.molecules[0], "morgan"):
                self.add_features(Features.morgan)

            fps = [mol.morgan for mol in self.molecules]

        dists = []
        nfps = len(fps)
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])

        self.clusters = Butina.ClusterData(
            dists, nfps, cutoff, isDistData=True
        )

        for id, cluster in enumerate(self.clusters):
            members = len(cluster)
            for i, mol_idx in enumerate(cluster):
                is_centroid = i == 0
                self.molecules[mol_idx].add_cluster_info(
                    id, members, is_centroid
                )
        return self.clusters

    def to_dataframe(self, *features):
        """Converts collection to df to build the model.
            If no features are supplied, all calculated features are added.
            (calls molecule.to_record())
        Returns
        -------
        df
            pd.DataFrame with all the information (clustering + features + activity)
        """

        if not hasattr(self, "clusters"):  # Cluster with default arguments if not clustered.
            self.cluster()

        records = list()
        for molecule in self.molecules:
            records.append(molecule.to_record(*features))
        df = pd.DataFrame(records)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float", errors="coerce")
        return df

    def plot_clustering_parameters(
        self, cutoffs=[0.2, 0.3, 0.4, 0.5, 0.6], thresholds=[10, 25, 50, 100],
        fp="topological"
    ):
        """Plots different combinations of parameters to asses clustering technique
        and decide outlier threshold.

        Parameters
        ----------
        cutoffs : list, optional
            Clutering distances to be plotted, by default [0.2, 0.3, 0.4, 0.5, 0.6]
        thresholds : list, optional
            Outlier thresholds to be added to the plots for visualization aid, 
            by default [10, 25, 50, 100]
        fp : str (topological or morgan), optional
            Type of fingerprints, by default "topological"

        Returns
        -------
        matplotlib.pyplot.figure
            Figure containing n plots, where n is the number of cutoffs supplied.
        """
        
        colors = ["#e3adb5", "#95b8e3", "#929195", "#53af8b", "r", "b"]
        fig = plt.figure(figsize=(12, 7))
        fig.tight_layout()
        for i, cutoff in enumerate(cutoffs):
            ax = fig.add_subplot(2, 3, i + 1)
            clusters = self.cluster(cutoff=cutoff, fp=fp)
            clusters_len = sorted([len(cluster) for cluster in clusters])
            N = len(clusters_len)
            ax.bar(range(N), clusters_len)
            ax.title.set_text(f"Cut off = {cutoff}")
            ax.set_xlabel("Cluster number")
            ax.set_ylabel("Cluster members")
            for i, th in enumerate(thresholds):
                if th < max(clusters_len):
                    y = th
                    n_mols = sum([x for x in clusters_len if x <= th])
                    max_val = max(clusters_len)
                    ax.axhline(
                        y=y, xmin=0, xmax=N, c=colors[i], linestyle="--",
                    )
                    ax.text(
                        x=0,
                        y=y + (max_val * 1 / 50),
                        s=f"{n_mols} molecules in clusters with ≤ {th} members",
                        color=colors[i],
                        fontsize="small"
                    )
        fig.tight_layout()
        return fig


class CollectionFromSDF(BaseCollection):
    """Builds collection reading SDF files.
    """

    def __init__(self, actives_sdf, inactives_sdf) -> None:
        """Reads SDF files and creates the molecules object.

        Parameters
        ----------
        actives_sdf : str
            Path to actives SDF.
        inactives_sdf : str
            Path to inactives SDF.
        """
        self.read_sdf(actives_sdf, inactives_sdf)
        self.molecules = self.actives + self.inactives

    def read_sdf(self, actives_sdf, inactives_sdf):
        """Reads SDF files with the rdkit.SDMolSupplier

        Parameters
        ----------
        actives_sdf : str
            Path to actives SDF.
        inactives_sdf : str
            Path to inactives SDF.
        """
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

