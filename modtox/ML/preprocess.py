import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from modtox.ML.descriptors_2D_ligand import *
from modtox.ML.external_descriptors import *
from rdkit.DataStructs import FingerprintSimilarity


TITLE_MOL = "molecules"
COLUMNS_TO_EXCLUDE = [ "Lig#", "Title", "Rank", "Conf#", "Pose#"]
LABELS = "labels"

class ProcessorSDF():

    def __init__(self, fp, descriptors, MACCS, csv, columns, label, debug=False):
        self.fp = fp
        self.descriptors = descriptors
        self.MACCS = MACCS
        self.external_data = csv
        self.columns = columns
        self.fitted = False
        self.label = label
        self.debug = debug


    def similarity_filter(self, sdfs_to_compare=None, cut=0.7, *sdfs):

        assert sdfs_to_compare is not None, "Must provide comparison sdfs"
        print('Similarity filtering with', os.path.abspath(sdfs_to_compare[0]), os.path.abspath(sdfs_to_compare[1]))
        mol_ref = []
        for sdf in tqdm(sdfs_to_compare):
            mol_ref += [mol for mol in Chem.SDMolSupplier(sdf) if mol]
        fps_ref = [Chem.Fingerprints.FingerprintMols.FingerprintMol(x) for x in mol_ref]
        mols_filtered = []
        for sdf in sdfs:
            print('Sdf checked', os.path.abspath(sdf))
            mols_tar= np.array([mol for mol in Chem.SDMolSupplier(sdf)])
            fps_tar = [Chem.Fingerprints.FingerprintMols.FingerprintMol(x) for x in mols_tar]
            idxs = []
            for i,fp1 in tqdm(enumerate(fps_tar), total=len(fps_tar)):
                for fp2 in fps_ref:
                    if FingerprintSimilarity(fp1, fp2) > cut:
                        idxs.append(i)
                        break
            print('Detected ' , len(idxs), ' similarities')
            good_idxs = [i for i in range(len(mols_tar)) if i not in idxs]
            mols_filtered.append(mols_tar[good_idxs])
        
        return mols_filtered
 
    def _load_train_set(self, sdf_active, sdf_inactive, sdfs_to_compare):
        """
        Separate between train and test dataframe
    
        Input:
            :input_file: str
            :sdf_property: str
        Output:
            :xtrain: Pandas DataFrame with molecules for training
            :xtest: Pandas DataFrame with molecules for testing
            :ytrain: Pandas Dataframe with labels for training
            :ytest: Pandas DataFrame with labels for testing
        """
        assert sdf_active, "active file must be given for analysis"
        assert sdf_inactive, "inactive file must be given for analysis"
                             
        print(sdf_active, sdf_inactive)
        actives = [ mol for mol in Chem.SDMolSupplier(sdf_active) if mol ]
        inactives = [ mol for mol in Chem.SDMolSupplier(sdf_inactive) if mol ]
    
        self.n_initial_active = len([mol for mol in Chem.SDMolSupplier(sdf_active)])
        self.n_initial_inactive = len([mol for mol in Chem.SDMolSupplier(sdf_inactive)])
        print("Active, Inactive")
        print(self.n_initial_active, self.n_initial_inactive)
        
        self.n_final_active = len(actives)
        self.n_final_inactive = len(inactives)
        print("Read Active, Read Inactive")
        print(self.n_final_active, self.n_final_inactive)
         
        #Similarity filtering	
        if sdfs_to_compare is not None:
            
            actives, inactives = [*self.similarity_filter(sdfs_to_compare, 0.7, sdf_active, sdf_inactive)]
            self.n_filtered_actives = len(actives)
            self.n_filtered_inactives = len(inactives)
            
            print("Filtered Active, Inactive")
            print(self.n_filtered_actives, self.n_filtered_inactives)

        #Do not handle tautomers with same molecule Name
        self.mol_names = []
        actives_non_repited = []
        inactives_non_repited = []
        for mol in actives:
            mol_name = mol.GetProp("_Name")
            if mol_name not in self.mol_names:
                self.mol_names.append(mol_name)
                actives_non_repited.append(mol)
        for mol in inactives:
            mol_name = mol.GetProp("_Name")
            if mol_name not in self.mol_names:
                self.mol_names.append(mol_name)
                inactives_non_repited.append(mol)
        #keep balance sets

        balance = False
        if balance:
            #random balance
            randoms = np.random.choice(range(max(len(inactives_non_repited), len(actives_non_repited))), min(len(inactives_non_repited), len(actives_non_repited)), replace=False) 

            if len(inactives_non_repited) > len(actives_non_repited):
                inactives_non_repited = np.array(inactives_non_repited)[randoms]
                randoms = np.add(randoms, len(actives_non_repited))
                idxs = np.concatenate((list(range(len(actives_non_repited))), randoms))
            else:
                idxs = np.concatenate((randoms, list(range(len( actives_non_repited), len(self.mol_names)))))
                actives_non_repeated = actives_non_repited[randoms]
            self.mol_names= [mol for i, mol in enumerate(self.mol_names) if i in idxs]

        #Main Dataframe
        actives_df = pd.DataFrame({TITLE_MOL: actives_non_repited })
        inactives_df =  pd.DataFrame({TITLE_MOL: inactives_non_repited })
    
        actives_df[LABELS] = [1,] * actives_df.shape[0]
        inactives_df[LABELS] = [0,] * inactives_df.shape[0]
    
        molecules = pd.concat([actives_df, inactives_df])
    
        print("Non Repeated Active, Non Repeated Inactive")
        print(actives_df.shape[0], inactives_df.shape[0])
        print("Shape Dataset")
        print(molecules.shape[0])

        return molecules

    def _load_sdf(self, sdf):
        molecules = [ mol for mol in Chem.SDMolSupplier(sdf, removeHs=False) if mol ]
        w = Chem.SDWriter('data_to_predict.sdf')
        for m in molecules: w.write(m) 
        molecules_df = pd.DataFrame({TITLE_MOL: molecules })
        return molecules_df

    def _load_inchies(self, inchies):
        molecules = [Chem.MolFromInchi(inchi, removeHs=False) for inchi in inchies]
        molecules_df = pd.DataFrame({TITLE_MOL: molecules })
        return molecules_df
         

    def _retrieve_header(self, folder='.', exclude=COLUMNS_TO_EXCLUDE):
        headers = []
        #Return training headers
        headers_fp = []; headers_de = []; headers_maccs = []; headers_ext = []
        if self.fp:
            headers_fp = np.loadtxt(os.path.join(folder, "daylight_descriptors.txt"), dtype=np.str)
            headers.extend(headers_fp)
        if self.descriptors:
            headers_de = np.loadtxt(os.path.join(folder, "2D_descriptors.txt"), dtype=np.str)
            headers.extend(headers_de)
        if self.MACCS:
            headers_maccs = np.loadtxt(os.path.join(folder, "MAC_descriptors.txt"), dtype=np.str)
            headers.extend(headers_maccs)
        if self.external_data:
            headers_ext = list(pd.read_csv(self.external_data))[1:]
            headers.extend(headers_ext)
        # Remove specified headers
        headers_to_remove = [feature for field in exclude for feature in headers if field in feature ]
        for header in list(set(headers_to_remove)):
            headers.remove(header)
            headers_ext.remove(header)
        return headers, headers_fp, headers_ext, headers_de, headers_maccs
 
    def fit(self, sdf_active, sdf_inactive, sdfs_to_compare=None):
        #Only loads molecules as a df["molecules"] = [rdkit instaces]
        sdf_active = os.path.abspath(sdf_active)
        sdf_inactive = os.path.abspath(sdf_inactive)
        self.data = self._load_train_set(sdf_active, sdf_inactive, sdfs_to_compare)
        self.fitted = True
        return self

    def fit_sdf(self, sdf):
        self.data = self._load_sdf(sdf)
        self.fitted = True
        return self

    def fit_inchies(self, inchies):
        self.data = self._load_inchies(inchies)
        self.fitted = True
        return self

    def transform(self, folder, exclude=COLUMNS_TO_EXCLUDE):
        assert self.fitted, "Please fit the processor"
        # Excluding labels
        if not self.debug:
            X = self.data.iloc[:, :-1]
            y = np.array(self.data.iloc[:,-1])
        else: 
            X = self.data.iloc[:2,:-1]
            y = np.array(self.data.iloc[:2,-1])
        molecular_data = [ TITLE_MOL, ]; numeric_features = []; features = []
        if self.fp:
            numeric_features.extend('fingerprint')
            features.extend([('fingerprint', Fingerprints(folder))])
        if self.descriptors:
            numeric_features.extend('descriptors')
            features.extend([('descriptors', Descriptors(folder))])
        if self.MACCS:
            numeric_features.extend('fingerprintMACCS')
            features.extend([('fingerprintMACCS', Fingerprints_MACS(folder))])
        if self.external_data:
            numeric_features.extend(['external_descriptors'])
            features.extend([('external_descriptors', ExternalData(self.external_data, self.mol_names, exclude=exclude, folder=folder))])
        
        transformer = FeatureUnion(features)
        preprocessor = ColumnTransformer(transformers=[('mol', transformer, molecular_data)])
        pre = Pipeline(steps=[('transformer', preprocessor)])
        X_trans = pre.fit_transform(X)
        print(X_trans)
        np.save("X", X_trans)
        np.save("Y", y)
        return X_trans, y

    def fit_transform(self, sdf_active, sdf_inactive, sdfs_to_compare=None, folder='.'):
        return self.fit(sdf_active, sdf_inactive, sdfs_to_compare).transform(folder)

    def transform_mol(self, folder, exclude=COLUMNS_TO_EXCLUDE):
        assert self.fitted, "Please fit the processor"
        # Excluding labels
        X = self.data
        np.save("mols_sdf", X)
        molecular_data = [ TITLE_MOL, ]; numeric_features = []; features = []
        if self.fp:
            numeric_features.extend('fingerprint')
            features.extend([('fingerprint', Fingerprints(folder))])
        if self.descriptors:
            numeric_features.extend('descriptors')
            features.extend([('descriptors', Descriptors(folder))])
        if self.MACCS:
            numeric_features.extend('fingerprintMACCS')
            features.extend([('fingerprintMACCS', Fingerprints_MACS(folder))])
        if self.external_data:
            numeric_features.extend(['external_descriptors'])
            features.extend([('external_descriptors', ExternalData(self.external_data, self.mol_names, exclude=exclude, folder=folder))])

        transformer = FeatureUnion(features)
        preprocessor = ColumnTransformer(transformers=[('mol', transformer, molecular_data)])
        pre = Pipeline(steps=[('transformer', preprocessor)])
        X_trans = pre.fit_transform(X)
        np.save("X_sdf", X_trans)
        return X_trans


    def fit_tranform_modtox(self, sdf_active, sdf_inactive, folder='.'):
        return self.fit(sdf_active, sdf_inactive).transform_mol(folder)

    def fit_transform_sdf(self, sdf, folder='.'):
        return self.fit_sdf(sdf).transform_mol(folder)

    def fit_transform_inchies(self, inchies, folder='.'):
        return self.fit_inchies(inchies).transform_mol(folder)
        
    def sanitize(self, X, y, cv, feature_to_check='external_descriptors', folder='.'):
        # function to remove non-docked instances and prepare datasets to model
        assert feature_to_check, "Need to provide external data path"
        self.headers, self.headers_fp, self.headers_ext, self.headers_de, self.headers_maccs = self._retrieve_header(folder=folder)
        # sorting mol_names to coincide with the indexing of csv
        #ordering names and y so that they have the same order than in x! (X is order at this point!)

        if self.external_data:
            self.mol_names, y = zip(*sorted(zip(self.mol_names, y)))
        else:
            self.mol_names, X, y = zip(*sorted(zip(self.mol_names, X, y)))


        X = np.array(X)

        if self.debug: self.mol_names = self.mol_names[:2]
        molecules_to_remove = []
        if feature_to_check == 'external_descriptors':
            assert self.external_data, "Need to read external data"
            indices_to_check = [i for i, j in enumerate(self.headers) if j in self.headers_ext] 
            molecules_to_remove =  [mol for i, mol in enumerate(self.mol_names) if np.isnan(X[i,indices_to_check]).all()]
            print('Number of molecules removed because all values are NAN', len(molecules_to_remove))
        if feature_to_check == 'fingerprintMACCS': 
            indices_to_check = [i for i, j in enumerate(self.headers) if j in self.headers_maccs] 
            molecules_to_remove =  [mol for i, mol in enumerate(self.mol_names) if np.isnan(X[i,indices_to_check]).all()]
            print('Number of molecules removed because all values are NAN', len(molecules_to_remove))
        if feature_to_check == 'descriptors': 
            indices_to_check = [i for i, j in enumerate(self.headers) if j in self.headers_de] 
            molecules_to_remove =  [mol for i, mol in enumerate(self.mol_names) if np.isnan(X[i,indices_to_check]).all()]
            print('Number of molecules removed because all values are NAN', len(molecules_to_remove))
        if feature_to_check == 'fingerprint': 
            indices_to_check = [i for i, j in enumerate(self.headers) if j in self.headers_fp] 
            molecules_to_remove =  [mol for i, mol in enumerate(self.mol_names) if np.isnan(X[i,indices_to_check]).all()]
            print('Number of molecules removed because all values are NAN', len(molecules_to_remove))

        #we have to remove the first value of the headers_ext descriptors since is the index of the molecule (1,2,3,4,5...)
        if feature_to_check == 'external_descriptors':
            to_remove_index = len(self.headers_de) + len(self.headers_maccs) + len(self.headers_fp) 
            X = np.delete(X, to_remove_index, axis=1)
        mols_to_maintain = [mol for mol in self.mol_names if mol not in molecules_to_remove]
        indxs_to_maintain = [np.where(np.array(self.mol_names) == mol)[0][0] for mol in mols_to_maintain]
        indxs_removed = [i for i in range(len(y)) if i not in indxs_to_maintain]
        y_removed = np.array(y)[indxs_removed]
        X_removed = X[indxs_removed, :]
        print('Removed 0s: ', len([l for l in y_removed if l==0]))
        print('Removed 1s: ', len([l for l in y_removed if l==1]))
        labels = np.array(y)[indxs_to_maintain]
        self.y = pd.Series(labels)
        self.y = np.array(self.y)
        self.X = X[indxs_to_maintain, :]
        self.mol_names = mols_to_maintain

        ###### saving 
        mols_ordered = mols_to_maintain + molecules_to_remove
        np.save('MOL_NAMES_{}'.format(self.label), mols_ordered)
        ######

        n_active_corrected = len([label for label in self.y if label==1])
        n_inactive_corrected = len([label for label in self.y if label==0])
        if cv > n_inactive_corrected  or cv > n_active_corrected:
             cv = min([n_active_corrected, n_inactive_corrected])

        
        return self.X, self.y, self.mol_names, y_removed, X_removed, cv 


    def filter_features(self, X):
         
        if self.columns:
            user_indexes = np.array([self.headers.index(column) for column in self.columns], dtype=int)
            self.X = X[:, user_indexes]
            self.headers = np.array(self.headers)[user_indexes].tolist()
          
        return 


