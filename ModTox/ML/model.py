import matplotlib.cm as cm
import operator
from sklearn.model_selection import GridSearchCV
import collections
from sklearn.feature_selection import RFE
import argparse
import pandas as pd
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
import ModTox.ML.classifiers as cl

TITLE_MOL = "molecules"
COLUMNS_TO_EXCLUDE = [ "Lig#", "Title", "Rank", "Conf#", "Pose#" ]
LABELS = "labels"


class GenericModel():

    def __init__(self, active, inactive, clf, csv=None, test=None, pb=False, fp=False, descriptors=False, MACCS=True):
        self.active = active
        self.inactive = inactive
        self.clf = clf
        self.pb = pb
        self.fp = fp
        self.descriptors = descriptors
        self.MACCS = MACCS
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

    def feature_transform(self, X, exclude=COLUMNS_TO_EXCLUDE):
        molecular_data = [ TITLE_MOL, ]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy="constant")),
            ('scaler', StandardScaler())
                ])
        
        numeric_features = []
        features = []
        if self.pb:
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

        molec_transformer = FeatureUnion(features)

        preprocessor = ColumnTransformer(
            transformers=[
                    ('mol', molec_transformer, molecular_data)])
        
        pre = Pipeline(steps=[('transformer', preprocessor),
                                  ('scaler', numeric_transformer),
                                                       ])
        return pre.fit_transform(X)
            
        

    def fit_transform(self, load=False, grid_search=False, output_model=None, save=False, cv=None, output_conf="conf.png"):
        
    
        self.x_train_trans = self.feature_transform(self.features)
        
    
        np.random.seed(7)



        if grid_search:
            param_grid = {"C":[1,10,100,1000], "gamma" : [1,0.1,0.001,0.0001], "kernel": ["linear", "rbf"]}
            grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0)
            grid.fit(self.x_train_trans, self.labels)
            print("Best Parameters")
            print(grid.best_params_)

        cv = self.n_final_active if not cv else cv

        if type(self.clf) is list and len(self.clf) > 0:
            print("Stack model")
            last_clf = self.clf[-1]
            scaler = StandardScaler()
            preds = np.array([ cross_val_predict(c, self.x_train_trans, self.labels, cv=cv) for c in self.clf[:-1 ]])
            print(self.x_train_trans.shape, preds.T.shape)
            X = np.hstack( [self.x_train_trans, preds.T] )
            prediction = cross_val_predict(last_clf, scaler.fit_transform(X), self.labels, cv=cv)
            
        else:
            print("Normal model")
            prediction = cross_val_predict(self.clf, self.x_train_trans, self.labels, cv=cv)
    
        conf = confusion_matrix(self.labels, prediction)

        conf[1][0] += (self.n_initial_active - self.n_final_active)
        conf[0][0] += (self.n_initial_inactive - self.n_final_inactive)

        print("{} KFOLD Training Crossvalidation".format(cv))
	print(conf.T)

        md.conf(conf[1][1], conf[0][1], conf[0][0], conf[1][0], output=output_conf)

        if output_model:
            pickle.dump(model, open(output, 'wb'))

    def load_model(self, model_file):
        print("Loading model")
        return pickle.load(open(model_file, 'rb'))

    def save(self, output):
        print("Saving Model")
        pickle.dump(model, open(output, 'wb'))

    def test_importance(self, names, clf):
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2        
        from sklearn.preprocessing import QuantileTransformer
        test, label_test, train, label_train = [], [], [], []
        indexes = []
        for i, (m, label) in enumerate(zip(self.features.values, self.labels.values)):
            m = m[0]
            if m.GetProp("_Name") in names:
                test.append(m)
                label_test.append(label)
                indexes.append(i)
            else:
                train.append(m)
                label_train.append(label)

        X_train = self.feature_transform(pd.DataFrame({"molecules": np.hstack([train, test])}))
        X_train_pos = QuantileTransformer(output_distribution='uniform').fit_transform(X_train)
        X_test = X_train_pos[-2:, :]
        model = clf.fit(X_test, label_test)

        bestfeatures = SelectKBest(score_func=chi2, k=10)
        fit = bestfeatures.fit(X_train_pos, np.hstack([label_train, label_test]))
        dfscores = pd.DataFrame({"Score":fit.scores_})
        dfscores["header"] = self.retrieve_header()
        print(dfscores.nlargest(10,'Score'))
        print(model.predict(X_test))

        bestfeatures = SelectKBest(score_func=chi2, k=10)
        fit = bestfeatures.fit(X_test, label_test)
        dfscores = pd.DataFrame({"Score":fit.scores_})
        dfscores["header"] = self.retrieve_header()
        print(dfscores.nlargest(10,'Score'))
        print(model.predict(X_test))


            
        

    def retrieve_header(self, exclude=COLUMNS_TO_EXCLUDE):
        headers = []
        #Return training headers
        if self.pb:
            headers_pb = np.empty(0)
            if self.fp:
                headers_pb = np.hstack([headers_pb, np.loadtxt("daylight_descriptors.txt", dtype=np.str)])
            if self.descriptors:
                headers_pb = np.hstack([headers_pb, np.loadtxt("2D_descriptors.txt", dtype=np.str)])
            if self.MACCS:
                headers_pb = np.hstack([headers_pb, np.loadtxt("MAC_descriptors.txt", dtype=np.str)])
            headers.extend(headers_pb.tolist())
        if self.external_data:
            headers.extend(list(pd.DataFrame.from_csv(self.external_data)))
        # Remove specified headers
        headers_to_remove = [feature for field in exclude for feature in headers if field in feature ]
        for header in headers_to_remove: headers.remove(header)
        return headers
        
    def feature_importance(self, clf=None, cv=1, number_feat=5, output_features="important_fatures.txt"):
        print("Extracting most importance features")
        self.headers = self.retrieve_header()
        assert len(self.headers) == self.x_train_trans.shape[1], "Headers and features should be the same length \
            {} {}".format(len(self.headers), self.x_train_trans.shape[1])
        clf = cl.XGBOOST
        model = clf.fit(self.x_train_trans, self.labels)
        important_features = model.get_booster().get_score(importance_type='gain')
        important_features_sorted = sorted(important_features.items(), key=operator.itemgetter(1), reverse=True)
        important_features_name = [[self.headers[int(feature[0].strip("f"))], feature[1]] for feature in important_features_sorted]
        np.savetxt(output_features, important_features_name, fmt='%s')

    def plot_descriptor(self, descriptor):
        print("Plotting descriptor {}".format(descriptor))
        headers = self.retrieve_header()
        index = headers.index(descriptor)
        data = self.x_train_trans[:, index]
        fig, ax = plt.subplots()
        ax.hist(data)
        fig.savefig("{}_hist.png".format(descriptor))

def parse_args(parser):
    parser.add_argument('--active', type=str,
                        help='sdf file with active compounds')
    parser.add_argument('--inactive', type=str,
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
                        help='cross validation k folds', default=None)
    parser.add_argument('--features', type=int,
                        help='Number of important features to retrieve', default=5)
    parser.add_argument('--features_cv', type=int,
                        help='KFold when calculating important features', default=1)
    parser.add_argument('--descriptors', nargs="+", help="descriptors to plot", default=[])
    parser.add_argument('--classifier', type=str, help="classifier to use", default="svm")
    parser.add_argument('--test_importance', nargs="+", help="Name of Molecules to include on testing feature importance", default=[])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build 2D QSAR model')
    parse_args(parser)
    args = parser.parse_args()
    clf = svm.SVC(C=1, gamma=1, kernel="linear")
    model = GenericModel(args.active, args.inactive, clf, csv=args.external_data, test=args.test, pb=args.pb)
    if args.load:
        model_fitted = model.load(args.load)
    else:
        X_train = model.fit_transform(model.features)
        #model.feature_importance(output_features="lucia.txt")
    if args.save:
        model.save(args.save)
    if args.test:
        X_test = model.feature_transform(model.data_test)
        prediction = model_fitted.predict(X_test)
        np.savetxt("results.txt", prediction)
    if args.test_importance:
        model.test_importance(args.test_importance, clf)
