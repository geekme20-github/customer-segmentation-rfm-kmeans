import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load transaction data and parse dates."""
    df = pd.read_csv(path, encoding='ISO-8859-1', parse_dates=['InvoiceDate'], dayfirst=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean transactions:
    - drop missing CustomerID
    - remove cancellations (InvoiceNo starting with 'C')
    - remove non-positive Quantity or UnitPrice
    - compute TotalPrice and normalize CustomerID
    """
    df = df.copy()

    # Drop records without CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Ensure InvoiceNo string type and drop cancellations/refunds
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]

    # Remove non-positive quantities/prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Total price
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Normalize CustomerID as string
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)

    # Reset index
    df = df.reset_index(drop=True)
    return df
