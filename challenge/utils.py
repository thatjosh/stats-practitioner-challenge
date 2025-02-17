from stochastic.processes.continuous import FractionalBrownianMotion
import numpy as np
import pandas as pd
from hurst import compute_Hc
from abc import ABC, abstractmethod

class TimeSeriesDf(ABC):
    def __init__(self, df):
        self.df = df

    @abstractmethod
    def get_start_value(self):
        """Returns the start value used in the prediction of the fBM."""
        pass

    @abstractmethod
    def get_mu(self):
        """Returns the mean of the time series."""
        pass

    @abstractmethod
    def get_sigma(self):
        """Returns the s.d. of the time series."""
        pass

    @abstractmethod
    def get_series_for_hurst(self):
        pass

    @abstractmethod
    def create_df_obj(self):
        pass


    def get_df(self):
        return self.df
    
    def get_window_slice(self, window_size):
        return self.get_df().iloc[-window_size:]

class VixDf(TimeSeriesDf):
    """Implements the TimeSeriesDf class."""
    def get_start_value(self):
        return self.df['vol'].iloc[-1]
    
    def get_mu(self):
        return self.df['log_vol_diff'].mean()
    
    def get_sigma(self):
        return self.df['log_vol_diff'].std()

    def get_series_for_hurst(self):
        return self.df['log_vol']

    def create_df_obj(self, data_slice):
        return VixDf(data_slice)

class SnpDf(TimeSeriesDf):
    """Implements the TimeSeriesDf class."""
    def get_start_value(self):
        pass

    def get_mu(self):
        pass
    
    def get_sigma(self):
        pass

    def get_series_for_hurst(self):
        pass

    def create_df_obj(self, data_slice):
        return SnpDf(data_slice)

def simulate_fbm_ndays(
        train_data_obj: TimeSeriesDf, H: float, n_days=1
    ):
    # DF correction (account for day 0)
    time_points = n_days + 1 

    # Calculate drift (μ) and volatility (σ)
    mu = train_data_obj.get_sigma()
    sigma = train_data_obj.get_sigma()

    # Starting value
    S0 = train_data_obj.get_start_value()

    # Time points for the drift component
    time_points = np.arange(time_points)

    # Create the fBM model
    fbm = FractionalBrownianMotion(hurst=H, t=n_days)

    # Assume this returns a full path including time 0
    fbm_full = fbm.sample(n=n_days)
    simulated_vix = S0 * np.exp(mu * time_points + sigma * fbm_full)

    # Remove t=0
    return simulated_vix[1:]

def apply_rolling_function(
        data_obj: TimeSeriesDf, window_size: int, predict_days: int
    ):
    """Applies the simulation function to the last rolling window slice of the time series 
    and predicts for the specified number of days."""

    # Extract the last rolling window slice (last n days)
    window_slice = data_obj.get_window_slice(window_size)
    window_slice_obj = data_obj.create_df_obj(window_slice)

    # Compute h
    h, _, _ = compute_Hc(data_obj.get_series_for_hurst(), kind='change', simplified=True)

    # Run the simulation for the specified number of days
    return simulate_fbm_ndays(window_slice_obj, h, predict_days)

def apply_rolling_predictions_from_start(
        data_obj: TimeSeriesDf, 
        start_date: str, 
        window_size: int,
    ):
        """
        Applies the simulation function on rolling windows starting from a given start_date.
        
        Returns: the predictions as a pandas df.
        """
        # Ensure the series is sorted
        ts = data_obj.get_df().sort_index()

        # Find the integer index for start_date
        start_idx = ts.index.searchsorted(pd.Timestamp(start_date), side='left')    
        predictions = {}

        # Loop over the series starting from when the first full window is available
        for i in range(start_idx - window_size - 1, len(ts)):
            slice = ts.iloc[i - window_size + 1 : i]
            window_slice = data_obj.create_df_obj(slice)
            predictions[ts.index[i]] = apply_rolling_function(window_slice, window_size, 1)

        # Convert predictions to a DF
        pred_df = pd.DataFrame.from_dict(predictions, orient='index')
        pred_df.columns = ["predicted"]
        pred_df = pred_df.sort_index()
        return pred_df