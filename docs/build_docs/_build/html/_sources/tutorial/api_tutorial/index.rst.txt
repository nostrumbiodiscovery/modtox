From python API
=================



Get your dataset from sdf files
---------------------------------------


Here we extract the input features from a serie of
active/inactive compounds. The available input features are:

- csv: Any external information with the name of the molecules as row index and the feature names as column indexes.

- MACCS: Generate MACCS fingerprints 

- fp: Generate Daylight fingerprints 

- descriptors: Generate topological descriptors

.. code-block:: python

    import os
    import numpy as np
    import modtox.ML.preprocess as Pre
    import modtox.ML.postprocess as Post
    import modtox.ML.model2 as model
    from sklearn.model_selection import train_test_split
    from sklearn.datasets.samples_generator import make_blobs
    import matplotlib.pyplot as plt

    folder = "tests_2/data/"
    sdf_active = os.path.join(folder, "actives.sdf")
    sdf_inactive = os.path.join(folder, "inactives.sdf")

    pre = Pre.ProcessorSDF(csv=csv, fp=False, descriptors=False, MACCS=True, columns=None)
    print("Fit and tranform for preprocessor..")
    X, y = pre.fit_transform(sdf_active=sdf_active, sdf_inactive=sdf_inactive)


Use your own dataset
----------------------

On the contrary here we use any external X, y dataset you could have

.. code-block:: python

    X, y = make_blobs(n_samples=100, centers=2, n_features=2,
                 random_state=0)

Curate dataset
-----------------

Drop samples with all Nans (sanitize) and remove the specified features (filter).
To specify the columns to remove use the columns argument on the model as:


pre = Pre.ProcessorSDF(csv=csv, fp=False, descriptors=False, MACCS=True, **columns=["Feature_1", "Feature_2"]**)

.. code-block:: python

    print("Sanitazing...")
    pre.sanitize(X, y)
    print("Filtering features...")
    pre.filter_features(X)

Fit model
--------------------------------------

You can choose between single/stack model as you want to use a stack of 5 classifiers
or only one.

.. code-block:: python

    Model = model.GenericModel(clf='stack', tpot=True)
    Model = model.GenericModel(clf='single', tpot=True)
    print("Fitting model...")
    Model.fit(X_train,y_train)

Predict
----------

.. code-block:: python

    y_pred = Model.predict(X_test)

Analysis
-----------

.. code-block:: python

    pp = Post.PostProcessor('stack', x_test=Model.X_test_trans, y_true_test=Model.Y_test,
                        y_pred_test=Model.prediction_test, y_proba_test=Model.predictions_proba_test,
                        x_train=Model.X_trans, y_true_train=Model.Y)

Metrics
****************

Plot the ROC and PR curve together
with the confusion matrix

.. code-block:: python

    ROC = pp.ROC()
    PR = pp.PR()
    DA = pp.conf_matrix()

Feature importance
***********************

Plot the features importance coming
from the shap values or he XGBOOST gain function

.. code-block:: python

    SH = pp.shap_values(debug=True)
    FI = pp.feature_importance()


Uncertanties
***************

Analyse the uncertanties of the model on the test samples

.. code-block:: python

    DA = pp.domain_analysis()
    UN = pp.calculate_uncertanties()


Visualize
************

Visualize the dataset and the wrong samples

.. code-block:: python

    pp.UMAP_plot()
    pp.PCA_plot()
    pp.tsne_plot()
