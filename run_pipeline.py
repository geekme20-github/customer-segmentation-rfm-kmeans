"""Run the customer segmentation pipeline end-to-end and save outputs.
Outputs:
- outputs/rfm_profile.csv
- outputs/pca_clusters.png
- outputs/selection_plot.png
"""
import os
import numpy as np
import pandas as pd
from customer_segmentation.load_and_clean import load_data, clean_data
from customer_segmentation.rfm import compute_rfm
from customer_segmentation.transform import transform_rfm
from customer_segmentation.modeling import find_clusters, plot_selection, fit_kmeans
from customer_segmentation.profiling import assign_and_profile
from customer_segmentation.visualize import pca_plot

# Paths for the dataset and output directory
DATA_PATH = "Ds_project_Customer_Segmentation/Dataset/online_retail.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Loading data
print('Loading data...')
df = load_data(DATA_PATH)
print(f'Raw rows: {len(df):,}')

# Cleaning data
print('Cleaning data...')
df = clean_data(df)
print(f'Clean rows: {len(df):,}')

# Computing RFM
print('Computing RFM...')
rfm, snapshot = compute_rfm(df)
print(f'Customers: {len(rfm):,}')

# Transforming features
print('Transforming features...')
X_scaled, scaler, features = transform_rfm(rfm)

# Finding cluster selection metrics
print('Finding cluster selection metrics...')
ks, inertias, silhouettes = find_clusters(X_scaled, k_min=2, k_max=10)
plot_selection(ks, inertias, silhouettes, outpath=os.path.join(OUT_DIR, 'selection_plot.png'))

# Determining the optimal number of clusters
best_k = ks[int(np.nanargmax(silhouettes))]
print(f'Chosen k by silhouette: {best_k}')

# Fitting KMeans
print('Fitting KMeans...')
km, labels = fit_kmeans(X_scaled, best_k)

# Profiling clusters
print('Profiling clusters...')
rfm_labeled, profile = assign_and_profile(rfm, labels)
profile_path = os.path.join(OUT_DIR, 'rfm_profile.csv')
rfm_labeled.to_csv(os.path.join(OUT_DIR, 'rfm_customers.csv'), index=False)
profile.to_csv(profile_path, index=False)
print('\nCluster profile:')
print(profile.to_string(index=False))

# Saving PCA plot
print('Saving PCA plot...')
pca_plot(X_scaled, labels, outpath=os.path.join(OUT_DIR, 'pca_clusters.png'))

# Outputs saved in outputs/
print('\nOutputs saved in outputs/')
print('Done')
