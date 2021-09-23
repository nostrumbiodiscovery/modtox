from col import Collection
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import cross_val_score


class Model:
    collection: Collection
    model: sklearn
    
    models = {
        "knn": KNeighborsClassifier(n_neighbors=2, p=1, weights="distance"),
        "svc": SVC(C=0.1, class_weight="balanced", kernel="rbf", tol=0.01),
        "lr": LogisticRegression(
            C=0.01, class_weight="balanced", dual=False, penalty="l2", max_iter=5000
        ),
        "tree": DecisionTreeClassifier(
            class_weight="balanced",
            criterion="entropy",
            max_depth=3,
            min_samples_leaf=1,
            min_samples_split=12,
            splitter="random",
        ),
        "nb": BernoulliNB(alpha=0.001, fit_prior=False),
    }
    
    def __init__( self, model_name: str, collection: Collection, 
                        glide=True, mordred=True, topo=True) -> None:
        try:
            self.model = self.models[model_name]
        except KeyError:
            model_names = ", ".join(self.models.keys())
            raise ModelError(
                f"Model '{model_name}' not available in this application.\n" 
                f"Available models: {model_names}."
            )
        self.collection = collection
        self.X, self.y = collection.to_Xy(self.collection.molecules, glide=glide, mordred=mordred, topo=topo)
        self.X_ext, self.y_ext = collection.to_Xy(self.collection.external_set, glide=glide, mordred=mordred, topo=topo)
        
        self.summary = dict()

    def cross_validation_scoring(self, k=5):
        "Cross validate and return metrics"
        if k == "LOO":
            k = len(self.collection.molecules)

        scores = cross_val_score(self.model, self.X, self.y, cv=k)
        mean = "%.2f" % float(scores.mean())
        std = "%.2f" % float(scores.std())
        return mean, std

    @staticmethod
    def improve_model_params(model):
        pass    

    def validate(self):
        y_pred = self.model.predict(self.collection.external_set)

        return
    
    
