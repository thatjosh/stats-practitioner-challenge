import pandas as pd
import numpy as np

def preprocess_data(csv_path: str, qty: str, processed_qty: str) -> pd.DataFrame:
    """
    Loads a CSV containing at least a Date column and a Price column,
    sorts by Date (ascending), and computes log returns.

    qty: name of outcome variable, e.g. prices, 
    processed_qty: name of processed outcome variable, e.g. log returns
    """
    df = pd.read_csv(csv_path)
    df.columns = ['date', qty]
    df['date'] = pd.to_datetime(df['date'])

    # Set date as index, sort oldest to newest
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    # Compute log returns
    df[processed_qty] = np.log(df[qty]/ df[qty].shift(1))
    
    # Drop the first row with NaN (1 row becomes NaN as differences are taken)
    df.dropna(subset=[processed_qty], inplace=True)
    return df[[qty, processed_qty]]