import xgboost as xgb
import warnings
import numpy as np
from sklearn import svm
from pactools.grid_search import GridSearchCVProgressBar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.scorer import make_scorer
from tpot import TPOTClassifier

#parameters

parameters_tree = {'criterion':('gini', 'entropy'), 'splitter': ('best', 'random'), 'class_weight':(None, 'balanced'), 'max_depth': range(1, 11), 
                       'min_samples_split': range(2, 20, 2), 'min_samples_leaf': range(1, 22, 2)}
parameters_bern = {'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.], 'fit_prior': [True, False]}
parameters_kn = {'n_neighbors': list(range(1,10)) + list(range(10,100,5)), 'weights': ["uniform", "distance"], 'p': [1, 2] }
parameters_svm = {'kernel': ['rbf', 'linear', 'poly','sigmoid'],  'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 
                    'class_weight':[None, 'balanced'],   'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]}
parameters_log = {'penalty': ["l1", "l2"], 'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.], 'dual': [False], 
                     'class_weight':[None, 'balanced']}
parameters_xgb = {'n_estimators': [100], 'max_depth': range(1, 11), 'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.], 'subsample': np.arange(0.05, 1.01, 0.05),
                     'min_child_weight': range(1, 21), 'nthread': [1]}
parameters_vot = {'kn__n_neighbors': range(1, 101, 5),'lr__C': [1.0, 100.0], 'tre__min_samples_split': range(2, 20, 2), 'tree__criterion':('gini', 'entropy'), 'nb__alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.], 
'svm__kernel': ['rbf', 'linear', 'poly','sigmoid'], 'svm__C': [1.0, 100.0]}
#default classifiers

XGBOOST = [xgb.XGBClassifier(), parameters_xgb]
SVM = [svm.SVC(), parameters_svm]
LR = [LogisticRegression(), parameters_log]
KN = [KNeighborsClassifier(), parameters_kn]
TREE = [DecisionTreeClassifier(), parameters_tree] 
NB = [BernoulliNB(), parameters_bern]

VOT = [VotingClassifier(estimators=[('svm', SVM), ('kn', KN), ('lr', LR),
 ('tree', TREE), ('nb', NB)], voting='soft' ), parameters_vot]

#optimized clfs for glide

SVM_OPT = svm.SVC(C=0.1, class_weight='balanced', kernel='rbf', tol= 0.01)
KN_OPT = KNeighborsClassifier(n_neighbors=2, p=1, weights='distance')
LR_OPT = LogisticRegression(C=0.01,class_weight='balanced', dual=False, penalty='l2')
TREE_OPT = DecisionTreeClassifier(class_weight='balanced', criterion='entropy',max_depth=3, min_samples_leaf=1, min_samples_split=12,splitter='random')
NB_OPT = BernoulliNB(alpha=0.001, fit_prior=False)

CLFS_OPT = [SVM_OPT, KN_OPT, LR_OPT, TREE_OPT, NB_OPT, NB[1]]


# Make a custom metric function
def mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)


def optimize_clf(X,Y, stack, clf):
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    print(clf)
    #choosing stratified fraction of the total data to optimize hyperparameters
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)
    
    optimized = False 
    if stack:
        print('Stack')
        estimators = []
        for k,cl in enumerate(clf):
            if optimized: 
                estimator = CLFS_OPT[k]
            else:
                clas = cl[0]
                params = cl[1]
                gscv = GridSearchCV(clas, params, cv=5, scoring='f1', error_score=0.0, return_train_score=True, verbose=10)
                gscv.fit(X_test, Y_test)
                best_params = gscv.best_params_
                estimator = gscv.best_estimator_
                print('best params', best_params)

            estimators.append(estimator)
        return estimators

    else:
        if optimized: #svm
            estimator = TREE_OPT
        else:
            clas = clf[0]    
            params = clf[1]
    
            random_search = RandomizedSearchCV(clas, param_distributions=params, n_iter=100, scoring='f1', verbose=3, cv=5, return_train_score=True)
            random_search.fit(X_test, Y_test)
            estimator = random_search.best_estimator_
            best_params = random_search.best_params_
            print('best score', random_search.best_score_)
            print('best estimator', estimator)
       #     gscv = GridSearchCV(clas, params, cv=5, scoring='f1', error_score=0.0, return_train_score=True, verbose=10)
       #     gscv.fit(X_test, Y_test)
       #     best_params = gscv.best_params_
       #     estimator = gscv.best_estimator_
 
        print('best params', best_params)

        return estimator       
  

def retrieve_classifier(classifier, X=None, Y=None, tpot=False, scoring='balanced_accuracy', cv=5, fast=False, generations=None, random_state=42, population_size=None, model=None):
    if classifier == "xgboost":
        clf = XGBOOST
    elif classifier == "single":
        if tpot:
            clf = get_tpot_classifier(cv=cv, fast=fast, scoring=scoring, generations=generations,random_state=random_state, population_size=population_size, model=model)
        else:
            clf = LR
    elif classifier == "stack":
        if tpot:
            clf = get_tpot_classifiers(cv=cv, fast=fast, scoring=scoring, generations=generations, random_state=random_state, population_size=population_size)
        else:
           # clf = [SVM, XGBOOST, LR, TREE, NB, KN]
            clf = [SVM, KN, LR, TREE, NB, NB]
    else:
        clf = classifier
    return clf

def get_tpot_classifiers(scoring='balanced_accuracy', generations=3, random_state=12, population_size=10, cv=10, fast=False):

    if fast:
        tpot_conf = "TPOT light"
    else:
        tpot_conf = None
    if scoring == 'mcc':
        print('mcc scorer')
        scoring = make_scorer(mcc, greater_is_better=True)
    
    scoring1 = 'precision_weighted'   
    scoring2 = 'f1_micro'   
    scoring = 'f1_macro'

    pipeline_optimizer1 = TPOTClassifier(scoring=scoring1, generations=generations, population_size=population_size, cv=cv,
                                        random_state=random_state, verbosity=2, config_dict=tpot_conf)
    pipeline_optimizer2 = TPOTClassifier(scoring=scoring2, generations=generations, population_size=population_size, cv=cv,
                                        random_state=random_state*2, verbosity=2, config_dict=tpot_conf)
    pipeline_optimizer3 = TPOTClassifier(scoring=scoring1, generations=generations, population_size=population_size, cv=cv,
                                        random_state=random_state*3, verbosity=2, config_dict=tpot_conf)
    pipeline_optimizer4 = TPOTClassifier(scoring=scoring2, generations=generations, population_size=population_size, cv=cv,
                                        random_state=1, verbosity=2, config_dict=tpot_conf)
    pipeline_optimizer5 = TPOTClassifier(scoring=scoring1, generations=generations, population_size=population_size, cv=cv,
                                        random_state=random_state*4, verbosity=2, config_dict=tpot_conf)
    pipeline_optimizer6 = TPOTClassifier(scoring=scoring, generations=generations, population_size=population_size, cv=cv,
                                        random_state=random_state*5, verbosity=2, config_dict=tpot_conf)
    return [pipeline_optimizer1, pipeline_optimizer2, pipeline_optimizer3, 
                    pipeline_optimizer4, pipeline_optimizer5, pipeline_optimizer6]

def get_tpot_classifier(scoring='balanced_accuracy', generations=20, population_size=10, cv=2, random_state=4212, fast=False, model=None, template=None):
    
    if fast:
        tpot_conf = "TPOT light"
        template = 'FeatureSetSelector-Transformer-Classifier'

    elif model =='kn':
        tpot_conf = {
        'sklearn.neighbors.KNeighborsClassifier': {
            'n_neighbors': range(1, 101),
            'weights': ["uniform", "distance"],
            'p': [1, 2]
          }
        }
    elif model =='nb':
        tpot_conf = {
        'sklearn.naive_bayes.BernoulliNB': {
            'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
            'fit_prior': [True, False]
           }
        }  

    elif model =='log':
        tpot_conf = {
        'sklearn.linear_model.LogisticRegression': {
            'penalty': ["l1", "l2"],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'dual': [True, False], 
            'class_weight': ['balanced']
          }
       }
    elif model == 'tree':
        tpot_conf = {
        'sklearn.tree.DecisionTreeClassifier': {
            'criterion': ["gini", "entropy"],
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21)
             } 
         }
    elif model == 'xgboost':
        tpot_conf = {
        'xgboost.XGBClassifier': {
            'n_estimators': [100],
            'max_depth': range(1, 11),
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'subsample': np.arange(0.05, 1.01, 0.05),
            'min_child_weight': range(1, 21),
            'nthread': [1], 
            'scale_pos_weight':[2]
            }
         }

    elif model == 'svm':
        tpot_conf = {
        'sklearn.svm.LinearSVC': {
            'class_weight':[None, 'balanced'],
            'penalty': ["l1", "l2"],
            'loss': ["hinge", "squared_hinge"],
            'dual': [True, False],
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    }
        }

    else:
        tpot_conf = {}

    if scoring == 'mcc':
        print('mcc scorer')
        scoring = make_scorer(mcc, greater_is_better=True)
 
    print(model, fast, scoring) 
    pipeline_optimizer = TPOTClassifier(scoring=scoring, generations=generations, population_size=population_size, cv=cv,
                                        random_state=random_state, verbosity=2, config_dict=tpot_conf, template=template)

    return pipeline_optimizer




