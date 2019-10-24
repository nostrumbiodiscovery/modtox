import matplotlib.pyplot as plt
import shap  # package used to calculate Shap values
import os
import operator
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier as RF

import modtox.ML.classifiers as cl

class PostProcessor():

    def __init__(self, x_test, y_true_test, y_pred_test, y_proba_test, x_train=None, y_true_train=None, folder='.'):
        
        self.x_train = x_train
        self.y_true_train = y_true_train

        self.x_test = x_test
        self.y_true_test = y_true_test
        self.y_pred_test = y_pred_test
        self.y_proba_test = y_proba_test

        self.folder = folder

    def ROC(self, output_ROC="ROC_curve.png"):
        roc_score = roc_auc_score(self.y_true_test, self.y_proba_test[:,1])
        fpr, tpr, threshold = roc_curve(self.y_true_test, self.y_proba_test[:,1]) #preds contains a tuple of probabilities for each 
        
        #plotting

        fig, ax = plt.subplots()

        ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_score)
        ax.legend(loc = 'lower right')
        ax.plot([0, 1], [0, 1],'r--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        fig.savefig(os.path.join(self.folder, output_ROC))
        plt.close()

        return roc_score

    def PR(self, output_PR="PR_curve.png"):

        precision, recall, thresholds = precision_recall_curve(self.y_true_test, self.y_proba_test[:,1])
        ap = average_precision_score(self.y_true_test, self.y_proba_test[:,1], average = 'micro')

        fig, ax = plt.subplots()

        ax.plot(recall, precision, alpha=0.2, color='b', label='AP = %0.2f' %ap)
        ax.legend(loc = 'lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        fig.savefig(os.path.join(self.folder, output_PR))
        plt.close()

        return ap

    def shap_values(self, output_shap='feature_importance_shap.png', names=None, debug=False, features=None):

        assert self.x_train.any() and self.y_true_train.any(), "Needed train and test datasets. Specify with x_train=X, ytrain=Y"

        names= ["sample_{}".format(i) for i in range(self.x_train.shape[0])] if not names else names
        features= ["feature_{}".format(i) for i in range(self.x_train.shape[1])] if not features else features

        clf = RF(random_state=213).fit(self.x_train, self.y_true_train) #now randomforest
        df = pd.DataFrame(self.x_test, columns = features)
        data_for_prediction_array = df.values
        clf.predict_proba(data_for_prediction_array)

        # Create object that can calculate shap values
        explainer = shap.TreeExplainer(clf)
        # Calculate Shap values
        shap_values = explainer.shap_values(data_for_prediction_array)[0]

        samples = names[0:1] if debug else names

        for row, name in enumerate(samples):
            shap.force_plot(explainer.expected_value[1], shap_values[row,:], df.iloc[row,:], matplotlib=True, show=False, text_rotation=90, figsize=(40, 10))
            plt.savefig(os.path.join(self.folder,'{}_shap.png'.format(name)))
        fig, axs = plt.subplots()
        shap.summary_plot(shap_values, df, plot_type="bar", show=False, auto_size_plot=True)
        fig.savefig(os.path.join(self.folder, output_shap) )
     

    def distributions(self, output_distributions = "distributions", features=None):

        assert self.x_train.any() and self.y_true_train.any(), "Needed train and test datasets. Specify with x_train=X, ytrain=Y"

        features= ["feature_{}".format(i) for i in range(self.x_train.shape[1])] if not features else features

        x_train_active = self.x_train[np.where(self.y_true_train == 1)]
        x_train_inactive = self.x_train[np.where(self.y_true_train == 0)]
        x_test_active = self.x_test[np.where(self.y_true_test == 1)]
        x_test_inactive = self.x_test[np.where(self.y_true_test == 0)]
    
        for x1, x2, name in [(self.x_train, self.x_test, 'dataset'), (x_train_active, x_test_active, 'active'), (x_train_inactive, x_test_inactive, 'inactive')]:
            for i in tqdm(range(len(x1[0]))):
                fig, ax1 = plt.subplots()
                sns.distplot(x1[:,i], label = 'Train set', ax=ax1)
                sns.distplot(x2[:,i], label = 'Test set', ax=ax1)
                plt.title('Train and test distributions')
                plt.savefig('{}/{}_{}_{}.png'.format(self.folder, output_distributions, name, i))
                plt.close()
 
    def feature_importance(self, clf=None, cv=1, number_feat=5, output_features="important_fatures.txt", features=None):
        
        print("Extracting most importance features")

        features= ["feature_{}".format(i) for i in range(self.x_train.shape[1])] if not features else features
        assert len(features) == self.x_test.shape[1], "Headers and features should be the same length \
            {} {}".format(len(features), self.x_test.shape[1])

        clf = cl.XGBOOST
        model = clf.fit(self.x_test, self.y_true_test)
        important_features = model.get_booster().get_score(importance_type='gain')
        important_features_sorted = sorted(important_features.items(), key=operator.itemgetter(1), reverse=True)
        important_features_name = [[features[int(feature[0].strip("f"))], feature[1]] for feature in important_features_sorted]
        np.savetxt(output_features, important_features_name, fmt='%s')
        features_name = [ feat[0] for feat in important_features_name ]

        return features_name

   # def tree_image():

