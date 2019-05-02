from sklearn.feature_selection import SelectFromModel



def filter_noisy_features(X, Y):
    """
        Filter noisy features
        by selecting the most correlated
        with the output vector
    """

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())
            ])

    ridge = linear_model.Ridge(alpha=150, solver='svd')
    
    clf_obj = ridge.fit(numeric_transformer.fit_transform(X), Y)
    
    model = SelectFromModel(clf_obj, prefit=True)
    
    X_new = model.transform(numeric_transformer.fit_transform(X))
    
    print("Inital dimnension {}, Final dimension {}".format(X.shape, X_new.shape))


def RFE_selection(X;Y)
