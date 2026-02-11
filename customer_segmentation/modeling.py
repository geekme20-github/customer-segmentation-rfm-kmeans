import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def find_clusters(X, k_min=2, k_max=10, random_state=42):
    ks = list(range(k_min, k_max+1))
    inertias = []
    silhouettes = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        # silhouette requires at least 2 clusters and less than n_samples
        try:
            s = silhouette_score(X, labels)
        except Exception:
            s = np.nan
        silhouettes.append(s)
    return ks, inertias, silhouettes


def plot_selection(ks, inertias, silhouettes, outpath=None):
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(ks, inertias, '-o', color='tab:blue', label='Inertia')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Inertia', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, '-s', color='tab:orange', label='Silhouette')
    ax2.set_ylabel('Silhouette', color='tab:orange')
    plt.title('Elbow (inertia) and Silhouette')
    if outpath:
        fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


def fit_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = km.fit_predict(X)
    return km, labels
