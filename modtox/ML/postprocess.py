import matplotlib.pyplot as plt
import shap  # package used to calculate Shap values
import os
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import modtox.ML.classifiers as cl
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier as RF
from scipy.spatial import distance

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


    def domain_analysis(self, output_densities="thresholds_vs_density.png", output_thresholds="threshold_analysis.txt", output_distplots="displot", debug=False, names=None):

        assert self.x_train.any() and self.y_true_train.any(), "Needed train and test datasets. Specify with x_train=X, ytrain=Y"

        names= ["sample_{}".format(i) for i in range(self.x_test.shape[0])] if not names else names
 
        ##### computing thresholds #######
        print("Computing applicability domains")

        distances = np.array([distance.cdist([x], self.x_train) for x in self.x_train])
        distances_sorted = [np.sort(d[0]) for d in distances]
        d_no_ii = [ d[1:] for d in distances_sorted] #discard 0 for the ii
        k = int(round(pow(len(self.x_train), 1/3)))

        d_means = [np.mean(d[:k][0]) for d in d_no_ii] #medium values
        Q1 = np.quantile(d_means, .25)
        Q3 = np.quantile(d_means, .75)
        IQR = Q3 - Q1
        d_ref = Q3 + 1.5*(Q3-Q1) #setting the refference value
        n_allowed =  []
        all_allowed = []
        for i in d_no_ii:
            d_allowed = [d for d in i if d <= d_ref]
            all_allowed.append(d_allowed)
            n_allowed.append(len(d_allowed))

        #selecting minimum value not 0:
        min_val = [np.sort(n_allowed)[i] for i in range(len(n_allowed)) if np.sort(n_allowed)[i] != 0]
        #replacing 0's with the min val
        n_allowed = [n if n!= 0 else min_val[0] for n in n_allowed]
        all_d = [sum(all_allowed[i]) for i, d in enumerate(d_no_ii)]
        thresholds = np.divide(all_d, n_allowed) #threshold computation
        thresholds[np.isinf(thresholds)] = min(thresholds) #setting to the minimum value where infinity

        count_active = []; count_inactive = []
       
        #now computing distances from train to test
   
        d_train_test = np.array([distance.cdist([x], self.x_train) for x in self.x_test])
   
        for i in d_train_test: # for each sample
            idxs = [j for j,d in enumerate(i[0]) if d <= thresholds[j]] #saving indexes of training with threshold > distance
            count_active.append(len([self.y_true_train[i] for i in idxs if self.y_true_train[i] == 1]))
            count_inactive.append(len([self.y_true_train[i] for i in idxs if self.y_true_train[i] == 0]))

        n_insiders = np.array(count_active) + np.array(count_inactive)
        mean_insiders = np.mean(n_insiders)
        
        df = pd.DataFrame()
        df['Names'] = names
        df['Thresholds'] = n_insiders
        df['Active thresholds'] = count_active
        df['Inactive thresholds'] = count_inactive
        df['True labels'] = self.y_true_test
        df['Prediction'] = self.y_pred_test
        df['Prediction probability'] = np.array([pred_1[1] for pred_1 in self.y_proba_test])
        df = df.sort_values(by=['Thresholds'], ascending = False) 

        with open(os.path.join(self.folder, output_thresholds), "w") as f:
            f.write(df.to_string())


        #####   plotting results ######


        if not debug:

            print('Plotting')
            plt.scatter(n_allowed, thresholds, c=self.y_true_train)
            plt.xlabel('Density')
            plt.ylabel('Threshold')
            plt.savefig(output_densities)
            plt.close()
        
            for j in tqdm(range(len(names))):
                #plotting 
                fig, ax1 = plt.subplots()
                sns.distplot(distances[j], label = 'Distances to train molecules', ax=ax1)
                sns.distplot(thresholds, label = 'Thresholds of train molecules', ax=ax1)
                ax1.set_xlabel('Euclidean distances')
                ax1.set_ylabel('Normalized counts')
                ax1.legend()

                ax2 = ax1.twinx()

                ax2.axhline(mean_insiders, ls='--', color = 'r', label = 'Mean App. Domain Membership')
                ax2.axhline(n_insiders[j], ls = '-', color = 'g', label = 'App. Domain Membership')
                ax2.set_ylim(min(n_insiders), max(n_insiders))
                ax2.set_ylabel('App.Domain counts')
                fig.tight_layout()
                ax2.legend(loc = 'center right')
                plt.title('{}'.format(names[j]))
                plt.savefig(os.path.join(self.folder, '{}_{}.png'.format(output_distplots, names[j])))
                plt.close()

   # def tree_image():

