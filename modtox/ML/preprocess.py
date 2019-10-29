import pandas as pd
from rdkit import Chem
from modtox.ML.descriptors_2D_ligand import *
from modtox.ML.external_descriptors import *


TITLE_MOL = "molecules"
COLUMNS_TO_EXCLUDE = [ "Lig#", "Title", "Rank", "Conf#", "Pose#"]
LABELS = "labels"

class ProcessorSDF():

    def __init__(self, fp, descriptors, MACCS, app_domain, csv, columns, debug=False):
        self.fp = fp
        self.descriptors = descriptors
        self.MACCS = MACCS
        self.app_domain = app_domain
        self.external_data = csv
        self.columns = columns
        self.fitted = False
        self.debug = debug
 
    def _load_train_set(self, sdf_active, sdf_inactive):
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
    
        #Main Dataframe
        actives_df = pd.DataFrame({TITLE_MOL: actives_non_repited })
        inactives_df =  pd.DataFrame({TITLE_MOL: inactives_non_repited })
    
        actives_df[LABELS] = [1,] * actives_df.shape[0]
        inactives_df[LABELS] = [0,] * inactives_df.shape[0]
    
        molecules = pd.concat([actives_df, inactives_df])
    
        print("Non Repited Active, Non Repited Inactive")
        print(actives_df.shape[0], inactives_df.shape[0])
        print("Shape Dataset")
        print(molecules.shape[0])

        return molecules

    def _retrieve_header(self, exclude=COLUMNS_TO_EXCLUDE):
        headers = []
        #Return training headers
        headers_pb = np.empty(0)
        headers_ext = []
        if self.fp:
            headers_pb = np.hstack([headers_pb, np.loadtxt("daylight_descriptors.txt", dtype=np.str)])
        if self.descriptors:
            headers_pb = np.hstack([headers_pb, np.loadtxt("2D_descriptors.txt", dtype=np.str)])
        if self.MACCS:
            headers_pb = np.hstack([headers_pb, np.loadtxt("MAC_descriptors.txt", dtype=np.str)])
        headers.extend(headers_pb.tolist())
        if self.external_data:
            headers_ext = list(pd.read_csv(os.path.join("descriptors", self.external_data)))
        headers.extend(headers_ext)
        # Remove specified headers
        headers_to_remove = [feature for field in exclude for feature in headers if field in feature ]
        for header in list(set(headers_to_remove)):
            headers.remove(header)
            headers_ext.remove(header)
        return headers[1:], headers_pb, headers_ext[1:]
 
    def fit(self, sdf_active, sdf_inactive):
        #Only loads molecules as a df["molecules"] = [rdkit instaces]
        self.data = self._load_train_set(sdf_active, sdf_inactive)
        self.fitted = True
        return self

    def transform(self, exclude=COLUMNS_TO_EXCLUDE):
        assert self.fitted, "Please fit the processor"
        # Excluding labels
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:,-1]
        molecular_data = [ TITLE_MOL, ]; numeric_features = []; features = []
        if self.fp:
            numeric_features.extend('fingerprint')
            features.extend([('fingerprint', Fingerprints())])
        if self.descriptors:
            numeric_features.extend('descriptors')
            features.extend([('descriptors', Descriptors())])
        if self.MACCS:
            numeric_features.extend('fingerprintMACCS')
            features.extend([('fingerprintMACCS', Fingerprints_MACS())])
        if self.external_data:
            numeric_features.extend(['external_descriptors'])
            features.extend([('external_descriptors', ExternalData(self.external_data, self.mol_names, exclude=exclude))])
        if self.app_domain:
            #Look at the ExternalData or others to have a sense on how to do it
            numeric_features.extend(['app_domain'])
            features.extend([('app_domain', AppDomain(self.external_data, self.mol_names, exclude=exclude))])
        
        transformer = FeatureUnion(features)
        preprocessor = ColumnTransformer(transformers=[('mol', transformer, molecular_data)])
        pre = Pipeline(steps=[('transformer', preprocessor)])
        X_trans = pre.fit_transform(X)
        return X_trans, y

    def fit_transform(self, sdf_active, sdf_inactive):

        return self.fit(sdf_active, sdf_inactive).transform()
        
    def sanitize(self, X, y, cv=5, feature_to_check='external_descriptors'):
  
        # function to remove non-docked instances and prepare datasets to model
        assert feature_to_check, "Need to provide external data path"
        self.headers, self.headers_pb, self.headers_ext = self._retrieve_header()
        molecules_to_remove = []

        if feature_to_check == 'external_descriptors':
            assert self.external_data, "Need to read external data"
            indices_to_check = [i for i, j in enumerate(self.headers) if j in self.headers_ext] 
            molecules_to_remove =  [mol for i, mol in enumerate(self.mol_names) if np.isnan(X[i,indices_to_check]).all()]
        if feature_to_check == 'fingerprintMACCS': 
            indices_to_check = [i for i, j in enumerate(self.headers) if j in self.headers_pb] 
            molecules_to_remove =  [mol for i, mol in enumerate(self.mol_names) if np.isnan(X[i,indices_to_check]).all()]

        if not self.debug:
            mols_to_maintain = [mol for mol in self.mol_names if mol not in molecules_to_remove]
            indxs_to_maintain = [np.where(np.array(self.mol_names) == mol)[0] for mol in mols_to_maintain]
            labels = np.array(y)[indxs_to_maintain]
            self.y = pd.Series(np.stack(labels, axis=1)[0])
            self.X = np.array(X)[indxs_to_maintain, :]
            self.mol_names = mols_to_maintain
            n_active_corrected = len([label for label in self.y if label==1])
            n_inactive_corrected = len([label for label in self.y if label==0])
            if cv > n_inactive_corrected  or cv > n_active_corrected:
                 cv = min([n_active_corrected, n_inactive_corrected])
        
        return self.X, self.y, self.mol_names 

    def filter_features(self, X):
         
        if self.columns:
            user_indexes = np.array([self.headers.index(column) for column in self.columns], dtype=int)
            self.X = X[:, user_indexes]
            self.headers = np.array(self.headers)[user_indexes].tolist()
          
        return self.X, self.headers


