TITLE_MOL = "molecules"
COLUMNS_TO_EXCLUDE = [ "Lig#", "Title", "Rank", "Conf#", "Pose#"]
LABELS = "labels"
CLF = ["SVM", "XGBOOST", "KN", "TREE", "NB", "NB_final"]


class GenericModel(object):

    def __init__(self, clf, load=filename_model, tpot=False, cv=None, debug=False):
        self.X = None
        self.Y = None
        self.fit = False
        self.filename_model = filename_model
        self.tpot = tpot
        self.cv = self.n_final_active if not cv else cv
        self.clf = cl.retrieve_classifier(clf, self.tpot, cv=self.cv, fast=True)
        self.debug = debug


    def fit(self, x, y):
        self.X = x
        self.Y = y
        #All of them if stack...
        self.clf.fit(self.X, self.Y)
        self.fit = True

    def predict(self, X_test):
        assert self.fit, "Please fit the model first"
        return self.clf.predict(X_test)


        
