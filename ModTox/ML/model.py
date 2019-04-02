import matplotlib.cm as cm
import argparse
import pickle
from sklearn.metrics import confusion_matrix
from sklearn import svm
import sys
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ModTox.ML.descriptors_2D_ligand import *
from ModTox.ML.external_descriptors import *
from ModTox.docking.glide import analyse as md


TITLE_MOL = "molecules"
LABELS = "labels"


class GenericModel():

    def __init__(self, active, inactive, clf, csv=None, test=None):
        self.active = active
        self.inactive = inactive
        self.clf = clf
        self.external_data = csv
        self.test = test
        self.data = self._load_training_set()
        self.features = self.data.iloc[:, :-1]
        self.labels = self.data.iloc[:, -1]
        if self.test:
            self.data_test = self._load_test()
            
    def _load_test(self):
        test_molecules = [ mol for mol in Chem.SDMolSupplier(self.test) if mol ]
        test_df = pd.DataFrame({TITLE_MOL: test_molecules })
        return test_df
    
    def _load_training_set(self):
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
        actives = [ mol for mol in Chem.SDMolSupplier(self.active) if mol ]
        inactives = [ mol for mol in Chem.SDMolSupplier(self.inactive) if mol ]
            

        self.n_initial_active = len([mol for mol in Chem.SDMolSupplier(self.active)])
        self.n_initial_inactive = len([mol for mol in Chem.SDMolSupplier(self.inactive)])
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
    
        self.data = molecules

        print("Non Repited Active, Non Repited Inactive")
        print(actives_df.shape[0], inactives_df.shape[0])

        print("Shape Dataset")
        print(self.data.shape[0])
    
        return self.data

    def feature_transform(self, X, pb=False):
        molecular_data = [ TITLE_MOL, ]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler())
                ])
        
        if pb:
            numeric_features = ['fingerprint', 'fingerprintMACS', 'descriptors']
            features = [
                ('fingerprint', Fingerprints()),
                ('descriptors', Descriptors()),
                ('fingerprintMACS',Fingerprints_MACS()),
                        ]        
        else:
            numeric_features = []
            features = []

        if self.external_data:
            numeric_features.extend(['external_descriptors'])
            features.extend([('external_descriptors', ExternalData(self.external_data, self.mol_names))])

        molec_transformer = FeatureUnion(features)

        preprocessor = ColumnTransformer(
            transformers=[
                    ('mol', molec_transformer, molecular_data)])
        
        pre = Pipeline(steps=[('transformer', preprocessor),
                                  ('scaler', numeric_transformer),
                                                       ])
        return pre.fit_transform(X)
            
        

    def fit_transform(self, load=False, grid_search=False, output=None, save=False, cv=100, pb=False):
        
    
        x_train_trans = self.feature_transform(self.features, pb=pb)
        
    
        np.random.seed(7)



        if grid_search:
            param_grid = {"C":[1,10,100,1000], "gamma" : [1,0.1,0.001,0.0001], "kernel": ["linear", "rbf"]}
            grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0)
            grid.fit(x_train_trans, self.labels)
            print("Best Parameters")
            print(grid.best_params_)

        prediction = cross_val_predict(self.clf, x_train_trans, self.labels, cv=cv)
    
        conf = confusion_matrix(self.labels, prediction)

        conf[1][0] += (self.n_initial_active - self.n_final_active)
        conf[0][0] += (self.n_initial_inactive - self.n_final_inactive)

        print("100 KFOLD Training Crossvalidation")
	print(conf.T)

        md.conf(conf[1][1], conf[0][1], conf[0][0], conf[1][0])

        if output:
            pickle.dump(model, open(output, 'wb'))

    def load_model(self, model_file):
        return pickle.load(open(model_file, 'rb'))

    def save(self, output):
        pickle.dump(model, open(output, 'wb'))

def parse_args(parser):
    parser.add_argument('active', type=str, 
                        help='sdf file with active compounds')
    parser.add_argument('inactive', type=str,
                        help='sdf file with inactive compounds')
    parser.add_argument('--test', type=str,
                        help='sdf file with test compounds', default=None)
    parser.add_argument('--external_data', type=str,
                        help='csv with external data to add to the model', default="glide_features.csv")
    parser.add_argument('--load', type=str,
                        help='load model from file', default=None)
    parser.add_argument('--save', type=str,
                        help='save model to file', default=None)
    parser.add_argument('--pb', action="store_true",
                        help='Compute physic based model (ligand topology, logP, logD...) or just glide model')
    parser.add_argument('--cv', type=int,
                        help='cross validation k folds', default=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build 2D QSAR model')
    parse_args(parser)
    args = parser.parse_args()
    clf = svm.SVC(C=1, gamma=1, kernel="linear")
    model = GenericModel(args.active, args.inactive, clf, csv=args.external_data, test=args.test)
    if args.load:
        model_fitted = model.load(args.load)
    else:
        X_train = model.feature_transform(model.features, pb=args.pb)
        model_fitted =  clf.fit(X_train, model.labels)
    if args.save:
        model.save(args.save)
    X_test = model.feature_transform(model.data_test, pb=args.pb)
    prediction = model_fitted.predict(X_test)
    np.savetxt("results.txt", prediction)
