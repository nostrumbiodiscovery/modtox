from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
    
estimators = {
    "knn": KNeighborsClassifier(),
    "svc": SVC(),
    "lr": LogisticRegression(),
    "tree": DecisionTreeClassifier(),
    "nb": BernoulliNB(),
}

distributions = {
    "knn": {
        'n_neighbors': list(range(1,10)) + list(range(10,100,5)), 
        'weights': ["uniform", "distance"], 
        'p': [1, 2] 
    },

    "svc": {
        'kernel': ['rbf', 'linear', 'poly','sigmoid'],  
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 
        'class_weight':[None, 'balanced'],   
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

    "lr": {
        'penalty': ["l1", "l2"], 
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.], 
        'dual': [False], 
        'class_weight':[None, 'balanced']
    },

    "tree":  {
        'criterion':('gini', 'entropy'), 
        'splitter': ('best', 'random'), 
        'class_weight':(None, 'balanced'), 
        'max_depth': range(1, 11), 
        'min_samples_split': range(2, 20, 2), 
        'min_samples_leaf': range(1, 22, 2)
    },

    "nb": {
        'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100], 
        'fit_prior': [True, False]
    },
}

estims = [(name, clf) for name, clf in estimators.items()] # Transforms to tuple for VotingClassifier
params = {  # Transforms 'distributions' to {'knn__p': [1, 2] } for VotingClassifer
    f"{name}__{param}": params 
    for name, param_dict in distributions.items() 
    for param, params in param_dict.items() 
}

halving_dist = {  # Some parameters raise warnings (should be looked at). For Halving, errors are raised instead of warnings, so it had to be modified.
    "knn": {
        'n_neighbors': list(range(1,10)) + list(range(10,100,5)), 
        'weights': ["uniform", "distance"], 
        'p': [1, 2] 
    },

    "svc": {
        'kernel': ['rbf', 'linear', 'poly','sigmoid'],  
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 
        'class_weight':[None, 'balanced'],   
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

    "lr": {
        'penalty': ['l2', 'none'], 
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.], 
        'dual': [False], 
        'class_weight':[None, 'balanced']
    },

    "tree":  {
        'criterion':('gini', 'entropy'), 
        'splitter': ('best', 'random'), 
        'class_weight':(None, 'balanced'), 
        'max_depth': range(1, 11), 
        'min_samples_split': range(2, 20, 2), 
        'min_samples_leaf': range(1, 22, 2)
    },

    "nb": {
        'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100], 
        'fit_prior': [True, False]
    },
}
