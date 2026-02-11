import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def pca_plot(X_scaled, labels, outpath=None):
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X_scaled)
    dfp = pd.DataFrame(proj, columns=['PC1','PC2'])
    dfp['Cluster'] = labels.astype(str)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='tab10', data=dfp, ax=ax, alpha=0.7)
    ax.set_title('PCA projection of clusters')
    ax.legend(title='Cluster')
    if outpath:
        fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)

