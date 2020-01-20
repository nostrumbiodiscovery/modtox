import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tpot.export_utils import generate_pipeline_code, expr_to_tree
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

    def __init__(self, X, Y, clf, filename_model='opt_model.pkl', folder='.',  tpot=False, cv=5, scoring='balanced_accuracy', generations=3, random_state=42, population_size=10, X_removed=None, y_removed=None, majvoting=False, weighting=False, debug=False):
        self.X = None
        self.Y = None
        self.fitted = False
        self.folder = folder
        self.filename_model = filename_model
        self.tpot = tpot
        self.y_removed = y_removed
        self.X_removed = X_removed
        self.cv = cv
        self.majvoting = majvoting
        self.weighting = weighting
        self.clf = cl.retrieve_classifier(clf,self.tpot, cv=self.cv, scoring=scoring, generations=generations, random_state=random_state, population_size=population_size, fast=False, model=None)
        self.random_state = random_state
        self.stack = self._is_stack_model()   

        self.scaler = StandardScaler()
        self.imputer = Imputer(imputer_type='simple')
        self.debug = debug

    def _is_stack_model(self):
        return type(self.clf) is list and len(self.clf) > 2

    def _extract_pred_proba(self, X, y, f=None, models = None):
        if f != None: 
            for cl in models:
                cl.fit(X, y)
                # 
                try: 
                    sklearn_pipeline_str = generate_pipeline_code(expr_to_tree(cl._optimized_pipeline, cl._pset), cl.operators)
                    sklearn_pipeline = eval(sklearn_pipeline_str, cl.operators_context)
                    cl._set_param_recursive(sklearn_pipeline.steps, 'random_state', self.random_state)
                except AttributeError:
                    pass
                # 
                if self.tpot: pickle.dump(sklearn_pipeline, f)
                else: pickle.dump(cl,f)

        if self.tpot:
            pred = np.array([cl.predict(X) for cl in models])
            try:
                proba = np.array([cl.predict_proba(X) for cl in models])
            except RuntimeError:
                proba = np.array([[[0,1] if pre == 1 else [1,0] for pre in pred[i]]for i in range(len(models))])
            except AttributeError:
                proba = np.array([[[0,1] if pre == 1 else [1,0] for pre in pred[i]]for i in range(len(models))])
        else:
            assert y.any(), "Need y"
            if self.fitted:
                proba = np.array([c.predict(X) for c in models])
            else:
                pred = np.array([ cross_val_predict(c, X, y, cv=self.cv) for c in models])
            for k, pp in enumerate(pred):
                cm = confusion_matrix(y, pp)
                print(models[k], cm)
            try:
                if self.fitted:
                    proba = np.array([c.predict_proba(X) for c in models])
                else:
                    proba = np.array([ cross_val_predict(c, X, y, cv=self.cv, method='predict_proba') for c in models])
            except AttributeError:
                proba = np.array([[[0,1] if pre == 1 else [1,0] for pre in pred[i]] for i in range(len(models))])
                
        return pred,proba
 
    def _stack(self, X, pred, proba, stack_type='proba'):

        if stack_type=='proba':
            return np.hstack( [X, np.transpose([z[:,0] for z in proba])])
        if stack_type=='label':
            return np.hstack( [X, pred.T])

    def _stack_final_results(self, y_indiv, y_final, Y_true):
        clf_result = np.vstack([y_indiv, y_final])
        return [([pred == true for pred, true  in zip(result, Y_true.tolist())]) for result in clf_result]

    def _last_fit(self, X, y, y_removed, f=None):
       if self.tpot:
           self.last_clf.fit(X,y)
           # 
           sklearn_pipeline_str = generate_pipeline_code(expr_to_tree(self.last_clf._optimized_pipeline, self.last_clf._pset), self.last_clf.operators)
           sklearn_pipeline = eval(sklearn_pipeline_str, self.last_clf.operators_context)
           self.last_clf._set_param_recursive(sklearn_pipeline.steps, 'random_state', self.random_state)
           # 
           prediction = self.last_clf.predict(X)
           try:
               prediction_proba = self.last_clf.predict_proba(X)
           except RuntimeError:
               prediction_proba = np.array([[0,1] if pred == 1 else [1,0] for pred in prediction])
               pass
           pickle.dump(sklearn_pipeline, f)
       else:
           prediction = cross_val_predict(self.last_clf, X, y, cv=self.cv)
           try:
               prediction_proba = cross_val_predict(self.last_clf, X, y, cv=self.cv, method='predict_proba')
           except AttributeError: 
               prediction_proba =  np.array([[0,1] if pred == 1 else [1,0] for pred in prediction])
           last_fitted = self.last_clf.fit(X,y)
           pickle.dump(last_fitted, f)

       if len(y_removed)>0:
           y_pred_removed = np.zeros((len(y_removed),), dtype=int)
           y_proba_removed = [[1,0] for i in range(len(y_removed))]
           prediction = np.concatenate((prediction, y_pred_removed))
           prediction_proba = np.concatenate((prediction_proba, y_proba_removed))

       return prediction, prediction_proba

    def _last_predict(self, X, y_removed):
        prediction = self.last_model.predict(X)
        try:
            proba = self.last_model.predict_proba(X)
        except AttributeError:
            proba = np.array([[0,1] if pred == 1 else [1,0] for pred in prediction])
            pass

        if len(y_removed)>0:
            y_pred_removed = np.zeros((len(y_removed),), dtype=int)
            y_proba_removed = [[1,0] for i in range(len(y_removed))]
            prediction = np.concatenate((prediction, y_pred_removed))
            proba = np.concatenate((proba, y_proba_removed))

        return prediction, proba

    def _f1_weigthing(self, y_true=None, y_preds=None):
      
      if not self.fitted:
          f1_scores = [f1_score(y_true, y_pred, average='weighted') for y_pred in y_preds]
          self.weight = f1_scores / np.sum(f1_scores)
      print('F1 weight', self.weight)
      y_weight =  np.array([x*y for x,y in zip(self.weight,y_preds)])
      y_weight = np.array([np.sum(x) for x in y_weight.T])
      y_weight_pred = np.array([[1-pred,pred] for pred in y_weight])
      return np.array([round(x) for x in y_weight]), y_weight_pred
      
    def _pipeline_fit(self, X, Y, y_removed, f):
        models = self.clf[:-1]
        self.indiv_fit, self.proba_fit =  self._extract_pred_proba(X, Y, f=f, models=models)
        if self.weighting:
            self.prediction_fit, self.prediction_proba_fit = self._f1_weigthing(Y, self.indiv_fit)       

        if self.majvoting: #majority voting
            print('Majvoting') 
            commons = [np.bincount(x) for x in self.indiv_fit.T]
            self.prediction_fit = np.array([np.argmax(i) for i in commons])
            self.prediction_proba_fit = np.array([[max(i)/sum(i),min(i)/sum(i)] for i in commons]) #now proba is just disagreement between clfs

        if not self.weighting and not self.majvoting:
            X_stack = self._stack(X, self.indiv_fit, self.proba_fit)
            self.prediction_fit, self.prediction_proba_fit = self._last_fit(X_stack, Y, f)
         
        #adding trivial predictions of 0's from nondocked molecules
        if len(y_removed)>0:
            y_pred_removed = np.zeros((len(y_removed),), dtype=int)  
            y_proba_removed = [[1,0] for i in range(len(y_removed))]  
            indiv_fit_removed = [y_pred_removed for i in range(len(models))]
            proba_fit_removed = [y_proba_removed for i in range(len(models))]
            self.prediction_fit = np.concatenate((self.prediction_fit, y_pred_removed))
            self.prediction_proba_fit = np.concatenate((self.prediction_proba_fit, y_proba_removed))        
            self.indiv_fit = [np.concatenate((x,y)) for x,y in zip(self.indiv_fit, indiv_fit_removed)] 
            Y = np.concatenate((Y, y_removed))

        self.clf_results = self._stack_final_results(self.indiv_fit, self.prediction_fit, Y)

    def _pipeline_predict(self, X, Y, y_removed):
        models = self.loaded_models
        if self.weighting:
            self.indiv_pred, self.proba_predict = self._extract_pred_proba(X, Y, models=models)
            self.prediction_test, self.predictions_proba_test = self._f1_weigthing(Y, self.indiv_pred)       

        if self.majvoting: #majority voting
            self.indiv_pred, self.proba_predict = self._extract_pred_proba(X, Y, models=models)
            commons = [np.bincount(x) for x in self.indiv_pred.T] 
            self.prediction_test = np.array([np.argmax(i) for i in commons])
            self.predictions_proba_test = np.array([[max(i)/sum(i),min(i)/sum(i)] for i in commons]) #now proba is just disagreement between clfs
        if not self.weighting and not self.majvoting:
            self.indiv_pred, self.proba_predict = self._extract_pred_proba(X, Y, models=models[:-1])
            X_pred_stack = self._stack(X, self.indiv_pred, self.proba_predict)
            self.prediction_test, self.predictions_proba_test = self._last_predict(X_pred_stack)

        if len(y_removed)>0:
            y_pred_removed = np.zeros((len(y_removed),), dtype=int)
            y_proba_removed = [[1,0] for i in range(len(y_removed))]
            indiv_pred_removed = [y_pred_removed for i in range(len(models))]
            proba_pred_removed = [y_proba_removed for i in range(len(models))]
            self.prediction_test = np.concatenate((self.prediction_test, y_pred_removed))
            self.predictions_proba_test = np.concatenate((self.predictions_proba_test, y_proba_removed))
            self.indiv_pred = [np.concatenate((x,y)) for x,y in zip(self.indiv_pred, indiv_pred_removed)] 
            Y = np.concatenate((Y, y_removed))

        self.clf_results =  self._stack_final_results(self.indiv_pred, self.prediction_test, Y)

    def load_models(self):
        print("Loading models")
        data = []
        with open(os.path.join(self.folder, self.train_folder, self.filename_model), 'rb') as rf:
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
        if len(self.X_removed) > 0:
            self.X_trans_removed = np.zeros(shape=self.X_removed.shape)

        if not self.tpot:
            print('Optimizing classifier')
            self.clf = cl.optimize_clf(self.X_trans,self.Y, self.stack, self.clf)

        if self.stack:
            self.last_clf = self.clf[-1]
            self._pipeline_fit(self.X_trans, self.Y, self.y_removed, f)
        else:
            self.last_clf = self.clf
            self.prediction_fit, self.prediction_proba_fit = self._last_fit(self.X_trans, self.Y, self.y_removed, f=f)
            self.indiv_fit=None
 
        self.fitted = True
        f.close()    
        self.Y = np.concatenate((self.Y, self.y_removed))
        if len(self.X_removed) > 0:
            self.X_trans = np.concatenate((self.X_trans, self.X_trans_removed))
        return self 

    def predict(self, X_test, Y_test, X_removed, y_removed, scaler=None, imputer=None, train_folder="."):
        assert self.fitted, "Please fit the model first"
        self.X_test = X_test
        self.Y_test = Y_test
        self.train_folder = train_folder
        if scaler != None: self.scaler = scaler
        if imputer != None: self.imputer = imputer
        self.X_test_trans = self.scaler.transform(self.imputer.transform(self.X_test))
        #for the removed molecules just imputing

        if len(X_removed)>0:
            X_trans_removed = np.zeros(shape=X_removed.shape)
        self.loaded_models = self.load_models()

        if self.stack:
            self.last_model = self.loaded_models[-1]
            self._pipeline_predict(self.X_test_trans, self.Y_test, y_removed)
            
        else:
            self.last_model = self.loaded_models[0]
            self.prediction_test, self.predictions_proba_test = self._last_predict(self.X_test_trans, y_removed)
            self.indiv_pred = None
        self.results = [ pred == true for pred, true in zip(self.prediction_test, self.Y_test)] #last classifier
        self.Y_test = np.concatenate((self.Y_test, y_removed))
        if len(X_removed)>0:
            self.X_test_trans = np.concatenate((self.X_test_trans, X_trans_removed))

        return self.results


