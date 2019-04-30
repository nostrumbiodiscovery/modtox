import umap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def UMAP_plot(X, Y, title="UMAP projection", fontsize=24, output="UMAPproj.png"):
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.2)
    embedding = reducer.fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[y] for y in np.array(Y)])
    fig.gca().set_aspect('equal', 'datalim')
    ax.set_title(title)
    fig.savefig(output)

def plot(X, Y, labels, title="plot", fontsize=24, output="proj.png", true_false=False):
    fig, ax = plt.subplots()
    if true_false:
        x_true = []
        y_true = []
        x_false = []
        y_false = []
        # Separate correct and incorrect
        for x, y, l in zip(X, Y, labels):
            if l:
                x_true.append(x)
                y_true.append(y)
            else:
                x_false.append(x)
                y_false.append(y)
        #Plot
        ax.scatter(x_true, y_true, c="g")
        ax.scatter(x_false, y_false, c="r")
    else:
        ax.scatter(X, Y, c=[sns.color_palette()[y] for y in np.array(labels)])
    ax.set_title(title)
    fig.savefig(output)
