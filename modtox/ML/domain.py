

def applicability_domain_model(column_to_transform="molecules"):
    """
    Model to detect and remove outlier samples

    Besd on the paper: ....................
    """

    molecular_data = [column_to_transform, ]
    
    numeric_features = ['sim', 'descriptors']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())
    ])
    
    molec_transformer = FeatureUnion([
        ('sim', Similarity_decomp()),
        ('descriptors', Descriptors()),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('mol', molec_transformer, molecular_data)])
    
    clf = Pipeline(steps=[('scaler', numeric_transformer),
                          ('clf', linear_model.Ridge(alpha=780, solver='svd', )),
                         ])
    
    X_DATA_TRANS = SimpleImputer().fit_transform(preprocessor.fit_transform(pd.concat([X_TRAIN,X_TEST])))
    Y_DATA_TRANS = pd.concat([Y_TRAIN,Y_TEST]).values

    return X_DATA_TRANS, Y_DATA_TRANS


def applicability_domain(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST,  X, Y, TRESHOLD=20):
    """
    Apply applicability domain
    into your dataset
    """

    THRESHOLD = 20
    
    n_samples, n_descriptors = np.shape(X_DATA_TRANS)

    #Mean of each descriptor
    means = np.array([ np.mean(X[0:111, i]) for i in range(n_descriptors)])
    
    #Std each descriptor
    stds = np.array([ np.std(X_DATA_TRANS[0:111, i]) for i in range(n_descriptors)])
    
    #Standarize values
    for i in range(n_descriptors):
        descriptor_values = X[:, i]
        for j, value in enumerate(descriptor_values):
            X[j][i] = (value - means[i])/stds[i]
            
    #If value bigger than value-mean > 3stds outside the box
    info = {}
    indexes = []
    for i in range(n_samples):
        inx = np.where( (X[i, :]>3) | (X[i, :]<-3))
        if inx[0].size > THRESHOLD:
            info[i]  = inx[0].size
            indexes.append(i)
    
            
    #Save numbers of data points removed
    indexes = np.array(indexes)
    removed_train = np.shape(indexes[np.where(indexes<111)])[0]
    
    #Molecules DATA X & Y
    DATA_X = pd.concat([X_TRAIN, X_TEST]).values
    DATA_Y = pd.concat([Y_TRAIN, Y_TEST]).values
    
    #Molecules with ouliers removed
    FINAL_DATA_X = np.delete(DATA_X, (list(info.keys())), axis=0)
    FINAL_DATA_Y = np.delete(DATA_Y, (list(info.keys())), axis=0)
    
    #Split in train and test again
    number_trainsamples = 111-removed_train
    X_TRAIN_FINAL = FINAL_DATA_X[0:number_trainsamples, :]
    X_TEST_FINAL = FINAL_DATA_X[number_trainsamples:, :]
    Y_TRAIN_FINAL = FINAL_DATA_Y[0:number_trainsamples, :]
    Y_TEST_FINAL = FINAL_DATA_Y[number_trainsamples:, :]
    
    return X_TRAIN_FINAL, X_TEST_FINAL, Y_TRAIN_FINAL, Y_TEST_FINAL 
