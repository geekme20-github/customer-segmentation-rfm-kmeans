import pandas as pd


def assign_and_profile(rfm, labels):
    df = rfm.copy()
    df['Cluster'] = labels
    profile = df.groupby('Cluster').agg(
        Count=('CustomerID','count'),
        Recency_mean=('RecencyDays','mean'),
        Frequency_mean=('Frequency','mean'),
        Monetary_mean=('Monetary','mean'),
        Recency_median=('RecencyDays','median'),
        Frequency_median=('Frequency','median'),
        Monetary_median=('Monetary','median')
    ).reset_index().sort_values('Monetary_mean', ascending=False)
    return df, profile
