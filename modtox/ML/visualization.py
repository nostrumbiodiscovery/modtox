import umap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.linalg import svd
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.offsetbox import AnchoredText

def ellipse_plot(position, width, height, angle, ax = None, dmin = 0.1, dmax = 0.5, n_ellipses=3, alpha=0.1, color=None):

    #ellipsoidal representation allow us to visualize 5-D data
    ax = ax or plt.gca()
    angle = (angle / np.pi) * 180
    # Draw the Ellipse
    for n in np.linspace(dmin, dmax, n_ellipses):
        ax.add_patch(Ellipse(position, n * width, n * height,
                             angle, alpha=alpha, lw=0, color=color))
def UMAP_plot(X, Y, title="UMAP projection", fontsize=24, output="UMAPproj.png"):
    colors = plt.get_cmap('Spectral')(np.linspace(0, 1, 2))
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.2, n_components = 5)
    embedding = reducer.fit_transform(X)
    fig, ax = plt.subplots()
    for i in range(embedding.shape[0]):
        pos = embedding[i, :2]
        Y = list(map(lambda x: int(x), Y)) # trues --> 1, falses ---> 0
        ellipse_plot(pos, embedding[i, 2],embedding[i, 3], embedding[i, 4], ax, dmin=0.2, dmax=1.0, alpha=0.03, color = colors[np.array(Y)[i]])
#    ax.scatter(embedding[:, 0], embedding[:, 1], c = [sns.color_palette()[y] for y in np.array(Y)])
    ax.scatter(embedding[:, 0], embedding[:, 1], c = Y, cmap = 'Spectral')
    fig.gca().set_aspect('equal', 'datalim')
    ax.set_title(title)
    fig.savefig(output)

def pca_plot(X, Y, title="PCA projection", output="PCAproj.png"):
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(X)
    variance_ratio = pca.explained_variance_ratio_
    variance_total = sum(variance_ratio)
    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[y] for y in np.array(Y)])
    anchored_text = AnchoredText('Ratio of variance explained: {}'.format(round(variance_total,2)), loc=2) # adding a box
    ax.add_artist(anchored_text)
    fig.gca().set_aspect('equal', 'datalim')
    ax.set_title(title)
    fig.savefig(output)

def tsne_plot(X, Y, title="TSNE projection", output="TSNEproj.png"):
    embedding = TSNE(n_components=2).fit_transform(X)
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
