from sklearn.dummy import DummyClassifier


def asses(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):
    
    for values in [ X_TRAIN, Y_TRAIN, X_TEST, Y_TEST ]:
        assert type(values) == np.ndarray, "{} must be array type".format(value)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())
            ])
    
    clf = Pipeline(steps=[('scaler', numeric_transformer),
                          ('clf', DummyClassifier()),
                                               ])
    
    asses(clf, preprocessor, X_TRAIN, np.ravel(Y_TRAIN), X_TEST, np.ravel(Y_TEST), val=True, test=True, plot=False, plot_test=False, number=50)
