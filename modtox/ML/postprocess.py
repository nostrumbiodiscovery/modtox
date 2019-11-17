import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import shap  # package used to calculate Shap values
import os
from matplotlib.lines import Line2D
import operator
import umap
import numpy as np
import pandas as pd
import seaborn as sns
import modtox.ML.classifiers as cl
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier as RF
from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import AnchoredText

class PostProcessor():

    def __init__(self, clf, x_test, y_true_test, y_pred_test, y_proba_test, y_pred_test_clfs=None, x_train=None, y_true_train=None, folder='.'):
        
        self.x_train = x_train
        self.y_true_train = y_true_train

        self.x_test = x_test
        self.y_true_test = y_true_test
        self.y_pred_test = y_pred_test
        self.y_proba_test = y_proba_test
        self.y_pred_test_clfs = y_pred_test_clfs
        self.clf = clf
        self.folder = folder
        if not os.path.exists(self.folder): os.mkdir(self.folder)

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

        assert self.x_train.any() and self.y_true_train.any(), "Needed train and test datasets. Specify with x_train=X, y_train=Y"
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

        for row, name in enumerate(tqdm(samples)):
            shap.force_plot(explainer.expected_value[1], shap_values[row,:], df.iloc[row,:], matplotlib=True, show=False, text_rotation=90, figsize=(40, 10))
            plt.savefig(os.path.join(self.folder,'{}_shap.png'.format(name)))
        fig, axs = plt.subplots()
        shap.summary_plot(shap_values, df, plot_type="bar", show=False, auto_size_plot=True)
        fig.savefig(os.path.join(self.folder, output_shap) )
     

    def distributions(self, output_distributions = "distributions", features=None, debug=False):

        assert self.x_train.any() and self.y_true_train.any(), "Needed train and test datasets. Specify with x_train=X, ytrain=Y"

        features= ["feature_{}".format(i) for i in range(self.x_train.shape[1])] if not features else features

        x_train_active = self.x_train[np.where(self.y_true_train == 1)]
        x_train_inactive = self.x_train[np.where(self.y_true_train == 0)]
        x_test_active = self.x_test[np.where(self.y_true_test == 1)]
        x_test_inactive = self.x_test[np.where(self.y_true_test == 0)]
    
        for x1, x2, name in [(self.x_train, self.x_test, 'dataset'), (x_train_active, x_test_active, 'active'), (x_train_inactive, x_test_inactive, 'inactive')]:
            if debug: end = range(3)
            else: end = range(len(x1[0]))
            for i in tqdm(end):
                fig, ax1 = plt.subplots()
                sns.distplot(x1[:,i], label = 'Train set', ax=ax1)
                sns.distplot(x2[:,i], label = 'Test set', ax=ax1)
                plt.title('Train and test distributions')
                plt.savefig('{}/{}_{}_{}.png'.format(self.folder, output_distributions, name, i))
                plt.close()
 
    def feature_importance(self, clf=None, cv=1, number_feat=5, output_features="important_features.txt", features=None):
        
        print("Extracting most importance features")

        features= ["feature_{}".format(i) for i in range(self.x_train.shape[1])] if not features else features
        assert len(features) == self.x_test.shape[1], "Headers and features should be the same length \
            {} {}".format(len(features), self.x_test.shape[1])

        clf = cl.XGBOOST
        model = clf.fit(self.x_test, self.y_true_test)
        important_features = model.get_booster().get_score(importance_type='gain')
        important_features_sorted = sorted(important_features.items(), key=operator.itemgetter(1), reverse=True)
        important_features_name = [[features[int(feature[0].strip("f"))], feature[1]] for feature in important_features_sorted]
        np.savetxt(os.path.join(self.folder, output_features), important_features_name, fmt='%s')
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
            count_active.append(len([self.y_true_train.tolist()[i] for i in idxs if self.y_true_train[i] == 1]))
            count_inactive.append(len([self.y_true_train.tolist()[i] for i in idxs if self.y_true_train[i] == 0]))

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
            plt.savefig(os.path.join(self.folder, output_densities))
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

    def conf_matrix(self, output_conf="confusion_matrix.png"):

        self.n_initial_active = len([np.where(self.y_true_test == 1)])
        self.n_initial_inactive = len([np.where(self.y_true_test == 0)]) #should be computed over the original set
        self.n_final_active = len([np.where(self.y_true_test == 1)]) 
        self.n_final_inactive = len([np.where(self.y_true_test == 0)])
        # Confusion Matrix
        conf = confusion_matrix(np.array(self.y_true_test), np.array(self.y_pred_test))
        conf[1][0] += (self.n_initial_active - self.n_final_active)
        conf[0][0] += (self.n_initial_inactive - self.n_final_inactive)
        df_cm = pd.DataFrame(conf, columns=np.unique(self.y_true_test), index = np.unique(self.y_true_test))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        print(df_cm)

        ax = sns.heatmap(data=df_cm, annot=True, cmap="Blues")
        plt.savefig(os.path.join(self.folder, output_conf))

    def ellipse_plot(self, position, width, height, angle, ax = None, dmin = 0.1, dmax = 0.5, n_ellipses=3, alpha=0.1, color=None):

        #ellipsoidal representation allow us to visualize 5-D data
        ax = ax or plt.gca()
        angle = (angle / np.pi) * 180
        # Draw the Ellipse
        for n in np.linspace(dmin, dmax, n_ellipses):
            ax.add_patch(Ellipse(position, n * width, n * height,
                                 angle, alpha=alpha, lw=0, color=color))

    def biplot_pca(self, score, coeff, folder=".", headers=None, labels=None):
        fig, ax = plt.subplots()
        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        plt.scatter(xs * scalex,ys * scaley, c=labels)
        importance = [x+y for x, y in zip(coeff[:,0], coeff[:,1])]
        indexes = np.argsort(importance)[::-1]
        headers = headers if headers else range(5)
        colors = ["r", "b", "y", "g", "m"]
        legend = []; custom_lines = []
        for i, c in zip(indexes[0:5], colors):
            plt.arrow(0, 0, coeff[i,0]*100, coeff[i,1]*100, color=c, alpha=0.5, width=0.005)
            legend.append(headers[i])
            custom_lines.append(Line2D([0], [0], color=c, lw=4))
        ax.set_xlabel("PC{}".format(1))
        ax.set_ylabel("PC{}".format(2))
        ax.legend(custom_lines, legend, loc="upper_left")
        fig.savefig(os.path.join(folder, "biplot.png"))


    def variance_plot(self):
        pca_tot = PCA()
        trans_X = pca_tot.fit_transform(self.x_test)
        variance_contributions = pca_tot.explained_variance_ratio_
        singular_values = normalize(pca_tot.singular_values_.reshape(1, -1), norm='l2').ravel()
        variance_explained = 0; j = 0; variance_vect = []; singular_values_chosen = []
        with open(os.path.join(self.folder, "Variances_values.txt"), 'w') as r:
            while variance_explained < 0.99:
                variance_explained +=  variance_contributions[j]
                variance_vect.append(variance_explained)
                singular_values_chosen.append(singular_values[j])
                r.write('{} component ---> Variance ratio: {} \n'.format(j+1, variance_explained))
                j+=1
    
        res = [x for x, val in enumerate(variance_vect) if val > 0.9] #list of indixes upper-90
        fig, ax = plt.subplots()
        ax.plot(range(j), variance_vect, c="y")
        ax.bar(range(j), singular_values_chosen)
        ax.axvline(x = res[0], ls = '--', c = 'r')
        ax.set_title('Variance vs Dimension')
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Variance ratio')
        fig.savefig(os.path.join(self.folder, 'Variance.png'))


    def PCA_plot(self, title="PCA projection", output_pca="PCA_proj.png", biplot=False):
    
        self.variance_plot()
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(self.x_test)
        if biplot:
            biplot_pca(embedding[:,0:2], np.transpose(pca.components_[0:2, :]), biplot, labels=self.y_true_test)
        variance_ratio = pca.explained_variance_ratio_
        variance_total = sum(variance_ratio)
        fig, ax = plt.subplots()
        ax.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[y] for y in np.array(self.y_true_test)])
        anchored_text = AnchoredText('Ratio of variance explained: {}'.format(round(variance_total,2)), loc=2) # adding a box
        ax.add_artist(anchored_text)
        fig.gca().set_aspect('equal', 'datalim')
        ax.set_title(title)
        fig.savefig(os.path.join(self.folder, output_pca))

    def tsne_plot(self, title="TSNE_projection", output_tsne="TSNE_proj.png"):
        embedding = TSNE(n_components=2).fit_transform(self.x_test)
        fig, ax = plt.subplots()
        ax.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[y] for y in np.array(self.y_true_test)])
        fig.gca().set_aspect('equal', 'datalim')
        ax.set_title(title)
        fig.savefig(os.path.join(self.folder, output_tsne))

    def calculate_uncertanties(self):
        if self.clf == 'stack':
            n_samples = len(self.y_pred_test_clfs[0])
            n_class_predicting_active = [0] * n_samples
            for pred in self.y_pred_test_clfs:
                for i, sample in enumerate(pred):
                    if sample == 1:
                        n_class_predicting_active[i] += 1
            n_classifiers = len(self.y_pred_test_clfs)
            uncertanties = [u/n_classifiers for u in n_class_predicting_active]
            return uncertanties
        else: print("uncertainties can't be computed in single model")


    def UMAP_plot(self, title="UMAP projection", fontsize=24, output_umap="UMAP_proj", single=False, wrong=False, wrongall=False, traintest=False, wrongsingle=False):

       reducer = umap.UMAP(n_neighbors=5, min_dist=0.2, n_components = 5)

       #getting the embedding
       if wrong or wrongall or traintest:  
           X = np.concatenate((self.x_train, self.x_test))
           embedding1 = reducer.fit_transform(X)
       if single: 
           X = self.x_test
           embedding2 = reducer.fit_transform(X)

       if traintest:
           # train and test separation
           embedding = embedding1
           colors = plt.get_cmap('Spectral')(np.linspace(0, 1, 2))
           fig, ax = plt.subplots()
           Y1 = list(map(lambda x: 0, self.y_true_train)) #setting 0 the train
           Y2 = list(map(lambda x: 1, self.y_true_test)) # setting 1 the test
           Y = np.concatenate((Y1,Y2))
           Y, idx = self._UMAP_sorting(X,Y)
           embedding = embedding[idx]
           for i in range(embedding.shape[0]):
               pos = embedding[i, :2]
               self.ellipse_plot(pos, embedding[i, 2],embedding[i, 3], embedding[i, 4], ax, dmin=0.2, dmax=1.0, alpha=0.01, color = colors[np.array(Y)[i]])
           end = self.x_train.shape[0]
           ax.scatter(embedding[:end, 0], embedding[:end, 1], c = 'b', cmap = 'Spectral', label= 'train')
           ax.scatter(embedding[end:, 0], embedding[end:, 1], c = 'r', cmap = 'Spectral', label= 'test')
           fig.gca().set_aspect('equal', 'datalim')
           plt.legend()
           ax.set_title(title)
           fig.savefig(os.path.join(self.folder, '{}_traintest.png'.format(output_umap)))
       if single: 
           # train and test reparation
           embedding = embedding2
           colors = plt.get_cmap('Spectral')(np.linspace(0, 1, 2))
           fig, ax = plt.subplots()
           Y = list(map(lambda x: int(x), self.y_true_test)) # trues --> 1, falses ---> 0
           Y, idx = self._UMAP_sorting(X,Y)
           embedding = embedding[idx]
           for i in range(embedding.shape[0]):
               pos = embedding[i, :2]
               self.ellipse_plot(pos, embedding[i, 2],embedding[i, 3], embedding[i, 4], ax, dmin=0.2, dmax=1.0, alpha=0.01, color = colors[np.array(Y)[i]])
           end = np.where(np.array(Y) == 0)[0][0]
           ax.scatter(embedding[:end, 0], embedding[:end, 1], c = 'g', cmap = 'Spectral', label='Active')
           ax.scatter(embedding[end:, 0], embedding[end:, 1], c = 'r', cmap = 'Spectral', label='Inactive')
           fig.gca().set_aspect('equal', 'datalim')
           plt.legend()
           ax.set_title(title)
           fig.savefig(os.path.join(self.folder, '{}_single.png'.format(output_umap)))
       if wrongsingle: 
           # train and test reparation
           embedding = embedding2
           colors = plt.get_cmap('Spectral')(np.linspace(0, 1, 2))
           fig, ax = plt.subplots()
           Y = list(map(lambda x: int(x), self.y_true_test)) # trues --> 1, falses ---> 0
           Yp = list(map(lambda x: int(x), self.y_pred_test)) # trues --> 1, falses ---> 0
           Y, idx = self._UMAP_sorting(X,Y)
           embedding = embedding[idx]
           Yp = np.array(Yp)[idx]
           for i in range(embedding.shape[0]):
               pos = embedding[i, :2]
               self.ellipse_plot(pos, embedding[i, 2],embedding[i, 3], embedding[i, 4], ax, dmin=0.2, dmax=1.0, alpha=0.01, color = colors[np.array(Y)[i]])
           end = np.where(np.array(Y) == 0)[0][0]
           ax.scatter(embedding[:end, 0], embedding[:end, 1], c = 'g', cmap = 'Spectral', label='Active')
           ax.scatter(embedding[end:, 0], embedding[end:, 1], c = 'r', cmap = 'Spectral', label='Inactive')
           mm = [ x == y for x,y in zip(Yp, Y)]
           indxs = np.array([ j for j, i in enumerate(mm) if i==False])
           indxs0 = [i for i in indxs if Yp[i] == 0]
           indxs1 = [i for i in indxs if Yp[i] == 1]
           if len(indxs0) >= 1:
               ax.scatter(embedding[indxs0, 0], embedding[indxs0, 1], c = 'y', cmap = 'Spectral', label= 'wrong as 0')
           if len(indxs1) >= 1:
               ax.scatter(embedding[indxs1, 0], embedding[indxs1, 1], c = 'y', marker='^', cmap = 'Spectral', label= 'wrong as 1')
           fig.gca().set_aspect('equal', 'datalim')
           plt.legend()
           ax.set_title(title)
           fig.savefig(os.path.join(self.folder, '{}_wrong_original.png'.format(output_umap)))
       if wrong:
           # train and test reparation
           embedding = embedding1
           colors = plt.get_cmap('Spectral')(np.linspace(0, 1, 4))
           fig, ax = plt.subplots()
           Y1 = list(map(lambda x: int(x) + 2, self.y_true_train))
           Y2 = list(map(lambda x: int(x), self.y_true_test))
           Y = np.concatenate((Y1,Y2))
           Y, idx = self._UMAP_sorting(X,Y)
           embedding = embedding[idx]

           # we need to reorder the test to choose the correct molecules

           Yp2 = list(map(lambda x: int(x), self.y_pred_test))
           Yp = np.concatenate((Y1,Yp2))
           Yp = Yp[idx]

           for i in range(embedding.shape[0]):
               pos = embedding[i, :2]
               self.ellipse_plot(pos, embedding[i, 2],embedding[i, 3], embedding[i, 4], ax, dmin=0.2, dmax=1.0, alpha=0.01, color = colors[np.array(Y)[i]])
           end = np.where(np.array(Y) == 2)[0][0]
           end1 = np.where(np.array(Y) == 1)[0][0]
           ax.scatter(embedding[:end, 0], embedding[:end, 1], c = 'b', cmap = 'Spectral', label= 'train active')
           ax.scatter(embedding[end:end1, 0], embedding[end:end1, 1], c = 'r', cmap = 'Spectral', label= 'train inactive')
           mm = [ x == y for x,y in zip(Yp, Y)]
           indxs = np.array([ j for j, i in enumerate(mm) if i==False])
           indxs0 = [i for i in indxs if Yp[i] == 0]
           indxs1 = [i for i in indxs if Yp[i] == 1]
           if len(indxs0) >= 1:
               ax.scatter(embedding[indxs0, 0], embedding[indxs0, 1], c = 'y', cmap = 'Spectral', label= 'wrong as 0')
           if len(indxs1) >= 1:
               ax.scatter(embedding[indxs1, 0], embedding[indxs1, 1], c = 'y', marker='^', cmap = 'Spectral', label= 'wrong as 1')
           fig.gca().set_aspect('equal', 'datalim')
           plt.legend()
           ax.set_title(title)
           fig.savefig(os.path.join(self.folder, '{}_wrong.png'.format(output_umap)))
 
       if wrongall:
           # wrong molecules and active/inactive for train and test and wrong mols
           embedding = embedding1
           colors = plt.get_cmap('Spectral')(np.linspace(0, 1, 4))
           fig, ax = plt.subplots()
           Y1 = list(map(lambda x: int(x) + 2, self.y_true_train))
           Y2 = list(map(lambda x: int(x), self.y_true_test))
           Y = np.concatenate((Y1,Y2))
           Y, idx = self._UMAP_sorting(X,Y)

           # we need to reorder the test to choose the correct molecules

           Yp2 = list(map(lambda x: int(x), self.y_pred_test))
           Yp = np.concatenate((Y1,Yp2))
           Yp = Yp[idx]
           embedding = embedding[idx]
           for i in range(embedding.shape[0]):
               pos = embedding[i, :2]
               self.ellipse_plot(pos, embedding[i, 2],embedding[i, 3], embedding[i, 4], ax, dmin=0.2, dmax=1.0, alpha=0.01, color = colors[np.array(Y)[i]])
           end = np.where(np.array(Y) == 2)[0][0]
           end1 = np.where(np.array(Y) == 1)[0][0]
           end2 = np.where(np.array(Y) == 0)[0][0]
           ax.scatter(embedding[:end, 0], embedding[:end, 1], c = 'b', cmap = 'Spectral', label= 'train active')
           ax.scatter(embedding[end:end1, 0], embedding[end:end1, 1], c = 'r', cmap = 'Spectral', label= 'train inactive')
           ax.scatter(embedding[end1:end2, 0], embedding[end1:end2, 1], c = 'g', cmap = 'Spectral', label= 'test active')
           ax.scatter(embedding[end2:, 0], embedding[end2:, 1], c = 'm', cmap = 'Spectral', label= 'test inactive')
           mm = [ x == y for x,y in zip(Yp, Y)]
           indxs = np.array([ j for j, i in enumerate(mm) if i==False])
           indxs0 = [i for i in indxs if Yp[i] == 0]
           indxs1 = [i for i in indxs if Yp[i] == 1]
           if len(indxs0) >= 1:
               ax.scatter(embedding[indxs0, 0], embedding[indxs0, 1], c = 'y', cmap = 'Spectral', label= 'wrong as 0')
           if len(indxs1) >= 1:
               ax.scatter(embedding[indxs1, 0], embedding[indxs1, 1], c = 'y', marker='^', cmap = 'Spectral', label= 'wrong as 1')
           fig.gca().set_aspect('equal', 'datalim')
           plt.legend()
           ax.set_title(title)
           fig.savefig(os.path.join(self.folder, '{}_wrong_all.png'.format(output_umap)))
 
    def _UMAP_sorting(self, X, Y):    
        # here we sort the values
        idx = list(range(len(Y)))
        Y, idx = (list(t) for t in zip(*sorted(zip(Y, idx), reverse=True)))
        return Y, idx
         
        
    def tree_image(): pass

