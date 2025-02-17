from stochastic.processes.continuous import FractionalBrownianMotion
import numpy as np
import pandas as pd
from hurst import compute_Hc

def simulate_n_days_for_vix(train_data, H=0.3, n_days=1):
    # DF correction (account for day 0)
    n_days += 1 

    # Calculate drift (μ) and volatility (σ)
    mu = train_data["log_vol_diff"].mean()
    sigma = train_data["log_vol_diff"].std()

    # Starting value
    S0 = train_data['vol'].iloc[-1]

    # Time points for the drift component
    time_points = np.arange(n_days)

    # Create the fBM model
    fbm = FractionalBrownianMotion(hurst=H, t=n_days-1)

    # Assume this returns a full path including time 0
    fbm_full = fbm.sample(n=n_days-1)

    # Include t0
    simulated_vix = S0 * np.exp(mu * time_points + sigma * fbm_full)
    # Remove t0
    return simulated_vix[1:]

def apply_rolling_function(df, col_name, window_size, predict_days):
    """Applies the simulation function to the last rolling window slice of the time series 
    and predicts for the specified number of days."""

    # Extract the last rolling window slice (last n days)
    window_slice = df.iloc[-window_size:]

    # Compute h
    h, _, _ = compute_Hc(window_slice[col_name], kind='change', simplified=True)
    # Run the simulation for the specified number of days
    return simulate_n_days_for_vix(window_slice, h, predict_days)

def apply_rolling_predictions_from_start(df, start_date, col_name, window_size, predict_days):
    """Applies the simulation function on rolling windows starting from a given start_date."""
    # Ensure the series is sorted
    ts = df.sort_index()

    # Find the integer index for start_date
    start_idx = ts.index.searchsorted(pd.Timestamp(start_date), side='left')    
    predictions = {}

    # Loop over the series starting from when the first full window is available
    for i in range(start_idx - window_size - 1, len(ts)):
        window_slice = ts.iloc[i - window_size + 1 : i]
        predictions[ts.index[i]] = apply_rolling_function(window_slice, col_name, window_size, 1)

    # Convert predictions to a DataFrame
    pred_df = pd.DataFrame.from_dict(predictions, orient='index')
    pred_df.columns = [f"predicted_{j+1}" for j in range(predict_days)]
    pred_df = pred_df.sort_index()
    return pred_df