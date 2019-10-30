import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import modtox.ML.classifiers as cl
import pandas as pd
import pickle


TITLE_MOL = "molecules"
COLUMNS_TO_EXCLUDE = [ "Lig#", "Title", "Rank", "Conf#", "Pose#"]
LABELS = "labels"
CLF = ["SVM", "XGBOOST", "KN", "TREE", "NB", "NB_final"]

class Imputer(object):

    def __init__(self, imputer_type, strategy='mean', fill_value=None, missing_values=np.nan, n_clusters=None):

        self.imputer_type = imputer_type
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values
        self.n_clusters = n_clusters
        self.imputer = None
        self._select_imputer()


    def _select_imputer(self):
    
        missing_values = self.missing_values 
        fill_value = self.fill_value
        strategy = self.strategy
        n_clusters = self.n_clusters

        if self.imputer_type == 'simple':
            self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value, missing_values=missing_values)
        if self.imputer_type == 'cluster_based':
            self.imputer = ImputerForSample(strategy=strategy, fill_value=fill_value, missing_values=missing_values, n_clusters=n_clusters)
        
    def fit(self, X):

        return self.imputer.fit(X)   

    def transform(self,X):

        return self.imputer.transform(X)

    def fit_transform(self, X):

        return self.imputer.fit_transform(X)


class ImputerForSample(object):
    
    def __init__(self, strategy='mean', fill_value=None, missing_values=np.nan, n_clusters = None):

        self.strategy = strategy
        self.fill_value = fill_value
        self.n_clusters = n_clusters
        self.missing_values = missing_values

    def fit(self, X):
         
        if self.strategy == 'constant':

            return    
    
        if self.strategy == 'mean':
    
            n_clusters = self.n_clusters
            
            assert isinstance(n_clusters, int) , "Must provide the number of clusters used"
            self.X_sep = np.array([np.split(x, self.n_clusters) for x in np.array(X)]) #splitting for cluster
            self.xmeans = np.array([np.nanmean(x_sep , axis=0) for x_sep in self.X_sep]) # mean without nan

            return self 

    def transform(self, X):
        
        if self.strategy == 'constant':

            fill_value = self.fill_value
            return np.array([ x if not np.isnan(x) else fill_value for x in X])

        if self.strategy == 'mean':
               #replace nan values by the mean of that feature (for each molecule) 
            xs = self.X_sep
            xm = self.xmeans
            xs = np.array([ [np.where(np.isnan(xs[j][i]), xm[j][i], xs[j][i]) for i in range(len(xs[j]))] for j in range(len(xs))])
            return np.array([np.concatenate(x) for x in xs])

    def fit_transform(self, X):

        return self.fit(X).transform(X)



class GenericModel(object):

    def __init__(self, clf, filename_model='opt_model.pkl', folder='.', tpot=False, cv=5, debug=False):
        self.X = None
        self.Y = None
        self.fitted = False
        self.folder = folder
        self.filename_model = filename_model
        self.tpot = tpot
        self.cv = cv
        self.clf = cl.retrieve_classifier(clf, self.tpot, cv=self.cv, fast=True)
        self.stack = self._is_stack_model()   

        self.scaler = StandardScaler()
        #self.imputer = Imputer(imputer_type='cluster_based', n_clusters=10)
        self.imputer = Imputer(imputer_type='simple')
        self.debug = debug

    def _is_stack_model(self):
        return type(self.clf) is list and len(self.clf) > 0

    def _extract_pred_proba(self, X, y, f=None, models = None):
        if f != None: 
            for cl in models:
                cl.fit(X, y)
                if self.tpot: pickle.dump(cl.fitted_pipeline_, f)
                else: pickle.dump(cl ,f)

        if self.tpot:
            pred = np.array([cl.predict(X) for cl in models])
            proba = np.array([cl.predict_proba(X) for cl in models])
        else:
            assert y.any(), "Need y"
            pred = np.array([ cross_val_predict(c, X, y, cv=self.cv) for c in models])
            proba = np.array([ cross_val_predict(c, X, y, cv=self.cv, method='predict_proba') for c in models])

        return pred,proba
 
    def _stack(self, X, pred, proba, stack_type='proba'):

        if stack_type=='proba':
            return np.hstack( [X, np.transpose([z[:,0] for z in proba])])
        if stack_type=='label':
            return np.hstack( [X, pred.T])

    def _stack_final_results(self, y_indiv, y_final, Y_true):
        clf_result = np.vstack([y_indiv, y_final])
        return [([pred == true for pred, true  in zip(result, Y_true.tolist())]) for result in clf_result]

    def _last_fit(self, X, y, f=None):

       if self.tpot:
           self.last_clf.fit(X,y)
           prediction = self.last_clf.predict(X)
           prediction_proba = self.last_clf.predict_proba(X)
           pickle.dump(self.last_clf.fitted_pipeline_, f)
       else:
           prediction = cross_val_predict(self.last_clf, X, y, cv=self.cv)
           prediction_proba = cross_val_predict(self.last_clf, X, y, cv=self.cv, method='predict_proba')
           last_fitted = self.last_clf.fit(X,y)
           pickle.dump(last_fitted, f)

       #pickle.dump(self.scaler, f)

       return prediction, prediction_proba

    def _last_predict(self, X):
        prediction = self.last_model.predict(X)
        proba = self.last_model.predict_proba(X)

        return prediction, proba


    def _pipeline_fit(self, X, Y, f):
        self.indiv_fit, proba =  self._extract_pred_proba(X, Y, f=f, models=self.clf[:-1 ])
        X_stack = self._stack(X, self.indiv_fit, proba)
        self.prediction_fit, self.prediction_proba_fit = self._last_fit(X_stack, Y, f)
        self.clf_results = self._stack_final_results(self.indiv_fit, self.prediction_fit, Y)

    def _pipeline_predict(self, X, Y):
  
        self.indiv_pred, proba = self._extract_pred_proba(X, Y, models=self.loaded_models[:-1])
        X_pred_stack = self._stack(X, self.indiv_pred, proba)
        self.prediction_test, self.predictions_proba_test = self._last_predict(X_pred_stack)
        self.clf_results =  self._stack_final_results(self.indiv_pred, self.prediction_test, Y)

    def load_models(self):
        print("Loading models")
        data = []
        with open(os.path.join(self.folder, self.filename_model), 'rb') as rf:
            try:
                while True:
                    data.append(pickle.load(rf))
            except EOFError:
                pass
        return data

    def fit(self, X, y):

        self.X = X
        self.Y = y
        f = open(os.path.join(self.folder, self.filename_model), 'wb')

        #imputing and scaling
        self.X_trans = self.scaler.fit_transform(self.imputer.fit_transform(self.X))        

        if self.stack:
            self.last_clf = self.clf[-1]
            self._pipeline_fit(self.X_trans, self.Y, f)
        else:
            self.last_clf = self.clf
            self.prediction_fit, self.prediction_proba_fit = self._last_fit(self.X_trans, self.Y, f=f)
        
        self.fitted = True
        f.close()    
        return self 

    def predict(self, X_test, Y_test):
        assert self.fitted, "Please fit the model first"
        
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_test_trans = self.scaler.transform(self.imputer.transform(self.X_test))
        self.loaded_models = self.load_models()

        if self.stack:
            self.last_model = self.loaded_models[-1]
            self._pipeline_predict(self.X_test_trans, self.Y_test)

        else:
            self.last_model = self.loaded_models
            self.prediction_test, self.predictions_proba_test = self._last_predict(self.X_test_trans)

        self.results = [ pred == true for pred, true in zip(self.prediction_test, self.Y_test)] #last classifier

        return self.results


