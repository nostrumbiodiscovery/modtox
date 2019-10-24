import modtox.helpers.loader

class ProcessorSDF():

    def __init__(self, fp, descriptors, MACCS, app_domain, csv, columns, imputer="constant", scaler="scaler"):
        self.fp = fp
        self.descriptors = descriptors
        self.MACCS = MACCS
        self.app_domain = app_domain
        self.external_data = csv
        self.columns = columns
    
    def fit(self, sdf):
        #Only loads molecules as a df["molecules"] = [rdkit instaces]
        self.data = loader._load_train(sdf)


    def transform(self):

        assert self.data, "Pleas fit the processor"

        X = [ TITLE_MOL, ]; numeric_features = []; features = []

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
            #Look at the ExternalData or others to have a sense on hwo to do it
            numeric_features.extend(['app_domain'])
            features.extend([('app_domain', AppDomain(self.external_data, self.mol_names, exclude=exclude))])
	

        transformer = FeatureUnion(features)
        preprocessor = ColumnTransformer(transformers=[('mol', transformer, X)])
        pre = Pipeline(steps=[('transformer', preprocessor)])
        X_trans = pre.fit_transform(X)
        return X_trans

    def fit_transform(self):
        return self.transform(self.fit(X))
        

   
