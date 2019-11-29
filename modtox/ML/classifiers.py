import xgboost as xgb
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from tpot import TPOTClassifier


XGBOOST = xgb.XGBClassifier(
learning_rate=0.01,
n_estimators=1000,
max_depth=4,
min_child_weight=6,
gamma=0,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27)


SVM = svm.SVC(C=1, gamma=0.1, kernel="rbf", probability=True)

KN = KNeighborsClassifier(7)

TREE = DecisionTreeClassifier(max_depth=5)

NB = GaussianNB()

def retrieve_classifier(classifier, tpot=False, cv=5, fast=False, generations=None, random_state=42, population_size=None):
    if classifier == "xgboost":
        clf = XGBOOST
    elif classifier == "single":
        if tpot:
            clf = get_tpot_classifier(cv=cv, fast=fast, generations=generations,random_state=random_state, population_size=population_size)
        else:
            clf = KN
    elif classifier == "stack":
        if tpot:
            clf = get_tpot_classifiers(cv=cv, fast=fast, generations=generations, random_state=random_state, population_size=population_size)
        else:
            clf = [SVM, XGBOOST, KN, TREE, NB, SVM]
    else:
        clf = classifier
    return clf

def get_tpot_classifiers(generations=3, random_state=12, population_size=10, cv=10, fast=False):

    if fast:
        tpot_conf = "TPOT light"
    else:
        tpot_conf = None

    pipeline_optimizer1 = TPOTClassifier(generations=generations, population_size=population_size, cv=cv,
                                        random_state=random_state, verbosity=2, config_dict=tpot_conf)
    pipeline_optimizer2 = TPOTClassifier(generations=generations, population_size=population_size, cv=cv,
                                        random_state=random_state*2, verbosity=2, config_dict=tpot_conf)
    pipeline_optimizer3 = TPOTClassifier(generations=generations, population_size=population_size, cv=cv,
                                        random_state=random_state*3, verbosity=2, config_dict=tpot_conf)
    pipeline_optimizer4 = TPOTClassifier(generations=generations, population_size=population_size, cv=cv,
                                        random_state=1, verbosity=2, config_dict=tpot_conf)
    pipeline_optimizer5 = TPOTClassifier(generations=generations, population_size=population_size, cv=cv,
                                        random_state=random_state*4, verbosity=2, config_dict=tpot_conf)
    pipeline_optimizer6 = TPOTClassifier(generations=generations, population_size=population_size, cv=cv,
                                        random_state=random_state*5, verbosity=2, config_dict=tpot_conf)
    return [pipeline_optimizer1, pipeline_optimizer2, pipeline_optimizer3, 
                    pipeline_optimizer4, pipeline_optimizer5, pipeline_optimizer6]

def get_tpot_classifier(generations=20, population_size=10, cv=2, random_state=4212, fast=False):

    if fast:
        tpot_conf = "TPOT light"
    else:
        tpot_conf = None

    pipeline_optimizer = TPOTClassifier(generations=generations, population_size=population_size, cv=cv,
                                        random_state=random_state, verbosity=2, config_dict=tpot_conf)

    return pipeline_optimizer

