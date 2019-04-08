import xgboost as xgb
from sklearn import svm


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


SVM = svm.SVC(C=1, gamma=1, kernel="linear")
