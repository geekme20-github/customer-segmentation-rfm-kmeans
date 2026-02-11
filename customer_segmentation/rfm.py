import pandas as pd


def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None):
    """Compute RFM at customer level.
    Recency: days since last purchase (snapshot - last InvoiceDate)
    Frequency: number of unique InvoiceNo per customer
    Monetary: sum of TotalPrice per customer
    Returns (rfm_df, snapshot_date)
    """
    # Ensure InvoiceDate is datetime
    df = df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    agg = df.groupby('CustomerID').agg(
        RecencyDays=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    agg['RecencyDays'] = agg['RecencyDays'].astype(int)
    return agg, snapshot_date
