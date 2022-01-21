import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm


def threshold(x_train, y_train, debug):

#as a first approach we set k = N**(1/3), being N the number of samples
#iterate over the full training set
# compute n-1 distances.
    distances = np.array([distance.cdist([x], x_train) for x in x_train])
    distances_sorted = [ np.sort(d[0]) for d in distances]
    d_no_ii = [ d[1:] for d in distances_sorted] #discard 0 for the ii
    k = int(round(pow(len(x_train), 1/3)))
    
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
    thres = np.divide(all_d, n_allowed) #threshold computation
    thres[np.isinf(thres)] = min(thres) #setting to the minimum value where infinity
    if not debug:
        print('Plotting')
        plt.scatter(n_allowed, thres, c=y_train)
        plt.xlabel('Density')
        plt.ylabel('Threshold') 
        plt.savefig('Threshold_vs_density.png')
    
    return thres


def evaluating_domain(xy_from_train, x1, y1, threshold, debug):
 
    #when running over the training set, we compute the domains internally;
    #whereas for the testing set is from test to train

    #splitting x and y from train:

    x_from_train = [x[0] for x in xy_from_train] #first component
    y_from_train = [x[1] for x in xy_from_train] #second component
 
    distances = np.array([distance.cdist([x], x_from_train) for x in x1]) #d between each test and training set
    n_insiders = []
    count_active = []
    count_inactive = []
    for i in distances: # for each molecule 
        idxs = [j for j,d in enumerate(i[0]) if d <= threshold[j]] #saving indexes of training with threshold > distance
        how_many_from_active =  len([y_from_train[i] for i in idxs if y_from_train[i] == 1]) #labels
        how_many_from_inactive =  len([y_from_train[i] for i in idxs if y_from_train[i] == 0])
        assert how_many_from_active + how_many_from_inactive == len(idxs)
        n_insiders.append(len(idxs))
        count_active.append(how_many_from_active)
        count_inactive.append(how_many_from_inactive)

    return n_insiders, count_active, count_inactive, distances

def analysis_domain(names, n_insiders, count_active, count_inactive, distances, labels, prediction, prediction_prob, threshold, debug):

    mean_insiders = np.mean(n_insiders)
  
    df = pd.DataFrame()
    df['Names'] = names
    df['Thresholds'] = n_insiders 
    df['Active thresholds'] = count_active
    df['Inactive thresholds'] = count_inactive
    df['True labels'] = labels
    df['Prediction'] = prediction
    df['Prediction probability'] = np.array([pred_1[1] for pred_1 in prediction_prob])
    
 
    df = df.sort_values(by=['Thresholds'], ascending = False)
  
    #plotting results 
    if not debug:
        if not os.path.exists("thresholds"): os.mkdir("thresholds")
        with open(os.path.join("thresholds", "threshold_analysis.txt"), "w") as f:
            f.write(df.to_string()) 
        for j in tqdm(range(len(names))):
            #plotting 
            fig, ax1 = plt.subplots()
            sns.distplot(distances[j], label = 'Distances to train molecules', ax=ax1)
            sns.distplot(threshold, label = 'Thresholds of train molecules', ax=ax1)
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
            plt.savefig('thresholds/distplot_{}.png'.format(names[j]))
            plt.close()
    return 


