import os
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import ensemble
import modtox.ML.postprocess as Post

def retrieve_processor(train_data=False):
    X, y = datasets.make_blobs(n_samples=1500, random_state=170, centers=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = ensemble.RandomForestClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    if train_data:
        return Post.PostProcessor(X_test, y_test, y_pred, y_proba, x_train=X_train, y_true_train=y_train)
    else:
        return Post.PostProcessor(X_test, y_test, y_pred, y_proba)


def clean(data):
    if isinstance(data, list):
        for i in data:
            if os.path.exists(i):
                os.remove(i)
    else:
        if os.path.exists(data):
            os.remove(data)

