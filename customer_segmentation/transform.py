import numpy as np
from sklearn.preprocessing import StandardScaler


def transform_rfm(rfm_df):
    df = rfm_df.copy()
    df['Monetary_log'] = np.log1p(df['Monetary'])
    df['Frequency_log'] = np.log1p(df['Frequency'])
    df['Recency_log'] = np.log1p(df['RecencyDays'])

    features = ['Recency_log', 'Frequency_log', 'Monetary_log']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled, scaler, features
