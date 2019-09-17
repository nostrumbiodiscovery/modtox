import matplotlib.cm as cm
import operator
from sklearn.model_selection import GridSearchCV
import collections
from sklearn.feature_selection import RFE
import argparse
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn import svm
import sys
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from modtox.ML.descriptors_2D_ligand import *
from modtox.ML.external_descriptors import *
from modtox.docking.glide import analyse as md
import modtox.ML.classifiers as cl
import modtox.ML.visualization as vs

TITLE_MOL = "molecules"
COLUMNS_TO_EXCLUDE = [ "Lig#", "Title", "Rank", "Conf#", "Pose#"]
LABELS = "labels"
CLF = ["SVM", "XGBOOST", "KN", "TREE", "NB", "NB_final"]


class GenericModel():

    def __init__(self, active, inactive, clf, csv=None, test=None, pb=False, fp=False, descriptors=False, MACCS=True, columns=None):
        self.active = active
        self.inactive = inactive
        self.clf = cl.retrieve_classifier(clf)
        self.pb = pb
        self.fp = fp
        self.descriptors = descriptors
        self.MACCS = MACCS
        self.external_data = csv
        self.columns = columns
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

    def __fit_transform__(self, X, exclude=COLUMNS_TO_EXCLUDE):
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
            
        

    def build_model(self, load=False, grid_search=False, output_model=None, save=False, cv=None, output_conf="conf.png", print_most_important=False):
        
    
        self.x_train_trans = self.__fit_transform__(self.features)

        self.headers = self.retrieve_header()
        
        #Filter columns
        if self.columns:
            user_indexes = np.array([self.headers.index(column) for column in self.columns], dtype=int)
            self.x_train_trans = self.x_train_trans[:, user_indexes]        
            self.headers = np.array(self.headers)[user_indexes].tolist()
        
    
        np.random.seed(7)
        cv = self.n_final_active if not cv else cv
        if type(self.clf) is list and len(self.clf) > 0:
            print("Stack model")
            last_clf = self.clf[-1]
            scaler = StandardScaler()
            #Predict with 5 classfiers
            preds = np.array([ cross_val_predict(c, self.x_train_trans, self.labels, cv=cv) for c in self.clf[:-1 ]])
            print(self.x_train_trans.shape, preds.T.shape)
            X = np.hstack( [self.x_train_trans, preds.T] )
            #Stack all classfiers in a final one
            prediction = cross_val_predict(last_clf, scaler.fit_transform(X), self.labels, cv=cv)
            prediction_prob = cross_val_predict(last_clf, scaler.fit_transform(X), self.labels, cv=cv, method='predict_proba')
            #Obtain results
            clf_result = np.vstack([preds, prediction])
            self.clf_results = [] # All classfiers
            for results in clf_result:
                self.clf_results.append([pred == true for pred, true  in zip(results, self.labels)])
            self.results = [ pred == true for pred, true in zip(prediction, self.labels)] #last classifier
            
        else:
            print("Normal model")
            prediction = cross_val_predict(self.clf, self.x_train_trans, self.labels, cv=cv)
            self.results = [ pred == true for pred, true in zip(prediction, self.labels)]

        # Plot Features
        vs.UMAP_plot(self.x_train_trans, self.labels, output="predictio_landscape.png")
        vs.UMAP_plot(self.x_train_trans, self.labels, output="sample_landscape.png")
        # Plot result each clf
        for result, clf_title  in zip(self.clf_results, CLF):
            vs.UMAP_plot(self.x_train_trans, result, output="{}.png".format(clf_title), title=clf_title)
        # Plot correlation matrice
        correlations = self.correlation_heatmap()
        #Plot features
        #self.plot_features([["Score", "Metal"],]) 


        #Report Errors
        print("\nMistaken Samples\n")
        errors = [ self.mol_names[i] for i, v in enumerate(self.results) if not v ]
        print(errors)

        # Retrieve list with feature importace
        print("\nImportant Features\n")
        important_features = self.feature_importance(clf=None, cv=1, number_feat=100, output_features="glide_features.txt")
        if print_most_important:
            print("\nMost Important Features\n")
            print(" ".join(important_features))
    
        # Confusion Matrix
        conf = confusion_matrix(self.labels, prediction)
        conf[1][0] += (self.n_initial_active - self.n_final_active)
        conf[0][0] += (self.n_initial_inactive - self.n_final_inactive)

        print("{} KFOLD Training Crossvalidation".format(cv))
        print(conf.T)

        md.conf(conf[1][1], conf[0][1], conf[0][0], conf[1][0], output=output_conf)

        # ROC CURVE
        import pdb; pdb.set_trace()
        
        self.plot_roc_curve_rate(self.labels, prediction_prob, prediction)

        if output_model:
            pickle.dump(model, open(output, 'wb'))

    def plot_features(self, plot_variables):
        """
        plot variables :: list of 2 fields [ ["Score", "Gscore"], ["Logp", "Logd"]]
        It will plot the two variables coloured by label
        """
        for field in plot_variables:
            name1 = field[0]
            name2 = field[1]
            idx1 = self.headers.index(name1)
            idx2 = self.headers.index(name2)
            values1 = self.x_train_trans[:, idx1]
            values2 = self.x_train_trans[:, idx2]
            vs.plot(values1, values2, self.labels, title="{}_{}_true_labels".format(name1, name2), output="{}_{}_true_labels.png".format(name1, name2))
            vs.plot(values1, values2, self.results, true_false=True, title="{}_{}_errors".format(name1, name2), output="{}_{}_errors.png".format(name1, name2))


    def plot_roc_curve(self, y_test, preds, n_classes=2):
        """
        Plot area under the curve
        """
        import scikitplot as skplt
        new_preds = []
        new_trues = []
        for p, t in zip(preds, y_test):
            if p == 0:
                new_preds.append([1, 0])
            else:
                new_preds.append([0, 1])
        skplt.metrics.plot_roc_curve(y_test.values, np.array(new_preds))
        plt.savefig("roc_curve.png")

    def plot_roc_curve_rate(self, y_test, preds, pred, n_classes=2):
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds[:,1])
        roc_auc = metrics.auc(y_test, pred)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        ax.legend(loc = 'lower right')
        ax.plot([0, 1], [0, 1],'r--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        fig.savefig("roc_curve.png")


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

        X_train = self.__fit_transform__(pd.DataFrame({"molecules": np.hstack([train, test])}))
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
        for header in list(set(headers_to_remove)): 
            headers.remove(header)
        return headers
        
    def feature_importance(self, clf=None, cv=1, number_feat=5, output_features="important_fatures.txt"):
        print("Extracting most importance features")
        assert len(self.headers) == self.x_train_trans.shape[1], "Headers and features should be the same length \
            {} {}".format(len(self.headers), self.x_train_trans.shape[1])
        clf = cl.XGBOOST
        model = clf.fit(self.x_train_trans, self.labels)
        important_features = model.get_booster().get_score(importance_type='gain')
        important_features_sorted = sorted(important_features.items(), key=operator.itemgetter(1), reverse=True)
        important_features_name = [[self.headers[int(feature[0].strip("f"))], feature[1]] for feature in important_features_sorted]
        np.savetxt(output_features, important_features_name, fmt='%s')
        features_name = [ feat[0] for feat in important_features_name ]
        return features_name

    def plot_descriptor(self, descriptor):
        print("Plotting descriptor {}".format(descriptor))
        headers = self.retrieve_header()
        index = headers.index(descriptor)
        data = self.x_train_trans[:, index]
        fig, ax = plt.subplots()
        ax.hist(data)
        fig.savefig("{}_hist.png".format(descriptor))

    def correlation_heatmap(self, output="correlation_map.png"):
        corr = pd.DataFrame(self.x_train_trans).corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,len(self.headers),1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.headers)
        ax.set_yticklabels(self.headers)
        fig.savefig(output)
        return corr

    def correlated_columns(self):
        corr_matrix = pd.DataFrame(self.x_train_trans).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [self.headers[column] for column in upper.columns if any(upper[column] > 0.95)]
        print(to_drop)

    def __predict__(self):
        print("Fitting...")
        #Initial Variables
        cv = self.n_final_active
        scaler = StandardScaler()
        #Preprocess data
        self.x_train_trans = self.__fit_transform__(self.features)
        print(self.x_train_trans)
        print("Size train", self.x_train_trans.shape)
        self.x_test_trans = self.__fit_transform__(self.data_test) 
        print(self.x_test_trans)
        print("Size test", self.x_test_trans.shape)
        # Fit pre-models
        self.models_fitted = [ c.fit(self.x_train_trans, self.labels) for c in self.clf[:-1 ]]
        #Predict pre-model
        preds_test = np.array([ model.predict(self.x_test_trans) for model in self.models_fitted ])
        # Fit last model
        preds_train = np.array([ cross_val_predict(c, self.x_train_trans, self.labels, cv=cv) for c in self.clf[:-1 ]])
        X = np.hstack( [self.x_train_trans, preds_train.T] )
        self.last_model = self.clf[-1].fit(scaler.fit_transform(X), self.labels)
        #Predict last model
        print(preds_test)
        X = np.hstack( [self.x_test_trans, preds_test.T] )
        prediction = self.last_model.predict(scaler.fit_transform(X))
        return prediction


def parse_args(parser):
    parser.add_argument('--active', type=str,
                        help='sdf file with active compounds')
    parser.add_argument('--inactive', type=str,
                        help='sdf file with inactive compounds')
    parser.add_argument('--test', type=str,
                        help='sdf file with test compounds', default=None)
    parser.add_argument('--external_data', type=str,
                        help='csv with external data to add to the model', default="glide_features.csv")
    parser.add_argument('--columns_to_keep', nargs="+", help="Name of columns to be kept from your external data", default=[])
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
    parser.add_argument('--print_most_important', action="store_true", help="Print most important features name to screen to use them as command lina arguments with --columns_to_keep")
    parser.add_argument('--build_model', action="store_true", help='Compute crossvalidation over active and inactives')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build 2D QSAR model')
    parse_args(parser)
    args = parser.parse_args()
    model = GenericModel(args.active, args.inactive, args.classifier, csv=args.external_data, test=args.test, pb=args.pb, columns=args.columns_to_keep)
    if args.load:
        model = model.load(args.load)
    if args.build_model:
        X_train = model.build_model(model.features, print_most_important=args.print_most_important)
        #model.feature_importance(output_features="lucia.txt")
    if args.save:
        model.save(args.save)
    if args.test:
        prediction = model.__predict__() 
        np.savetxt("results.txt", prediction)
    if args.test_importance:
        model.test_importance(args.test_importance, clf)
