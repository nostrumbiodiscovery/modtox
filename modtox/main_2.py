import os
import numpy as np
import modtox.ML.preprocess as Pre
import modtox.ML.postprocess as Post
import modtox.ML.model2 as model
from sklearn.model_selection import train_test_split


folder = "/home/moruiz/modtox_dir/modtox/tests_2/data"
sdf_active = os.path.join(folder, "actives.sdf")
sdf_inactive = os.path.join(folder, "inactives.sdf")
csv = os.path.join(folder, "glide_features.csv")
#preprocess

pre = Pre.ProcessorSDF(csv=csv, fp=False, descriptors=False, MACCS=True, columns=None, app_domain=False)
print("Fit and tranform for preprocessor..")
X, y = pre.fit_transform(sdf_active=sdf_active, sdf_inactive=sdf_inactive)
print("Sanitazing...")
pre.sanitize(X, y)
print("Filtering features...")
pre.filter_features(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#fit model
Model = model.GenericModel(clf='stack', tpot=True)
print("Fitting model...")
Model.fit(X_train,y_train)

#predict model
print("Predicting...")
y_pred = Model.predict(X_test, y_test)
print(y_pred)
