from stochastic.processes.continuous import FractionalBrownianMotion
import numpy as np
import pandas as pd
from hurst import compute_Hc
from abc import ABC, abstractmethod
from arch import arch_model

"""
Common utility methods demonstrated below:

#### 1. Predict 1 day ####
vix_training_data_obj = utils.VixDf(vix_training_df)
next_day_pred = utils.simulate_fbm(vix_training_data_obj, 0.3, 1)
print(f'Next day prediction: {next_day_pred}')

#### 2. Predict 5 days ####
vix_training_data_obj = utils.VixDf(vix_training_df)
utils.predict_with_fbm(vix_training_data_obj, 500, 5)
print(f'Next 5 days prediction: {next_day_pred}')

#### 3. Predict from a start date, for a number of days, and then computes RMSE ####
# let SNP be a df that's given, it should contain the price/vol, and its associated logged counterparts
snp_obj = utils.SnpDf(SNP)
res_df = utils.apply_rolling_predictions_from_start(snp_obj, '2020-01-01', 150)
res_df = res_df.join(SNP, how='inner')
utils.compute_rmse(res_df, 'predicted', 'price')
"""

def compute_rmse(df, predicted_col_name, actual_col_name):
    diff = df[predicted_col_name] - df[actual_col_name]
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    return rmse

class TimeSeriesDf(ABC):
    def __init__(self, df):
        self.df = df
        self.df.sort_index(ascending=True)

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
        """Gets the time series data required to calculate the hurst exp."""
        pass

    @abstractmethod
    def create_df_obj(self):
        """Creates a df object of the same type."""
        pass

    def get_df(self):
        return self.df
    
    def get_window_slice(self, window_size):
        return self.get_df().iloc[-window_size:]

class VixDf(TimeSeriesDf):
    """Implements the fBM model for S&P500 VIX data."""
    def get_start_value(self):
        return self.df['vol'].iloc[-1]
    
    def get_mu(self):
        return self.df['log_vol_diff'].mean()
    
    def get_sigma(self):
        return self.df['log_vol_diff'].std()

    def get_series_for_hurst(self):
        return self.df['log_vol_diff']

    def create_df_obj(self, data_slice):
        return VixDf(data_slice)

class SnpDf(TimeSeriesDf):
    """Implements the fBM model for S&P500 data."""
    def get_start_value(self):
        return self.df['price'].iloc[-1]
    
    def get_mu(self):
        return self.df['log_returns'].mean()
    
    def get_sigma(self):
        return self.df['log_returns'].std()

    def get_series_for_hurst(self):
        return self.df['log_returns']

    def create_df_obj(self, data_slice):
        return SnpDf(data_slice)

class NasdaqDf(TimeSeriesDf):
    """Implements the fBM model for Nasdaq data."""
    def get_start_value(self):
        return self.df['price'].iloc[-1]
    
    def get_mu(self):
        return self.df['log_returns'].mean()
    
    def get_sigma(self):
        return self.df['log_returns'].std()

    def get_series_for_hurst(self):
        return self.df['log_returns']

    def create_df_obj(self, data_slice):
        return NasdaqDf(data_slice)

def fit_garch_and_obtain_conditional_vol(log_returns: pd.Series):
    garch_model = arch_model(log_returns, vol='Garch', p=1, q=1)
    res = garch_model.fit(disp='off')
    conditional_volatilities = res.conditional_volatility
    return conditional_volatilities[-1]

class GarchFbmReturnForecast(TimeSeriesDf):
    """Implements the GARCH-fBM model with S&P500 / NASDAQ data."""
    def get_start_value(self):
        return self.df['price'].iloc[-1]
    
    def get_mu(self):
        return self.df['log_returns'].mean()
    
    def get_sigma(self):
        return fit_garch_and_obtain_conditional_vol(self.df['log_returns'])

    def get_series_for_hurst(self):
        return self.df['log_returns']

    def create_df_obj(self, data_slice):
        return GarchFbmReturnForecast(data_slice)
    
class GarchFbmVolForecast(TimeSeriesDf):
    """Implements the GARCH-fBM model with S&P500 VIX data."""
    def get_start_value(self):
        return self.df['vol'].iloc[-1]
    
    def get_mu(self):
        return self.df['log_vol_diff'].mean()
    
    def get_sigma(self):
        return fit_garch_and_obtain_conditional_vol(self.df['log_vol_diff'])

    def get_series_for_hurst(self):
        return self.df['log_vol_diff']

    def create_df_obj(self, data_slice):
        return GarchFbmVolForecast(data_slice)

def simulate_fbm(
        train_data_obj: TimeSeriesDf, H: float, n_days=1, n_simulations=1000
    ):
    # DF correction (account for day 0)
    time_points = n_days + 1 

    # Calculate drift (μ) and volatility (σ)
    mu = train_data_obj.get_mu()
    sigma = train_data_obj.get_sigma()

    # Starting value
    S0 = train_data_obj.get_start_value()

    # Create an array of time points for the drift component
    times = np.arange(time_points)

    # Initialize an array to store the simulated paths
    simulated_paths = np.zeros((n_simulations, len(times)))
    
    # Monte Carlo simulation loop: simulate n_paths of FBM
    for i in range(n_simulations):
        # FBM path (inc time 0)
        fbm = FractionalBrownianMotion(hurst=H, t=n_days)
        fbm_full = fbm.sample(n=n_days)
        simulated_data = S0 * np.exp(mu * times + sigma * fbm_full)
        simulated_paths[i, :] = simulated_data
    ensemble_average = np.mean(simulated_paths, axis=0)
    
    # Return the average outcomes for days 1 through n_days (excluding t=0)
    return ensemble_average[1:]

def predict_with_fbm(
        data_obj: TimeSeriesDf, window_size: int, predict_days: int
    ):
    """Applies the simulation function to the last rolling window slice of the time series 
    and predicts for the specified number of days."""

    # Extract the last rolling window slice (last n days)
    window_slice = data_obj.get_window_slice(window_size)
    window_slice_obj = data_obj.create_df_obj(window_slice)

    # Compute hurst exponent
    h, _, _ = compute_Hc(data_obj.get_series_for_hurst(), kind='change', simplified=True)

    # Run simulation
    return {
        'fbm': simulate_fbm(window_slice_obj, h, predict_days),
        'conditional_vol': window_slice_obj.get_sigma()
    }

def apply_rolling_predictions_from_start(
        data_obj: TimeSeriesDf, 
        start_date: str, 
        window_size: int,
    ) -> pd.DataFrame:
        """Applies the simulation function on rolling windows starting from a given start_date.
        Returns: the predictions as a pandas df.
        """
        # Ensure the series is sorted
        ts = data_obj.get_df().sort_index()

        # Find the integer index for start_date
        start_idx = ts.index.searchsorted(pd.Timestamp(start_date), side='left')    
        dates, predictions, cond_vols = [], [], []

        # Loop over the series starting from when the first full window is available
        for i in range(start_idx - window_size - 1, len(ts)):
            print(f'Day {i}...')
            slice = ts.iloc[i - window_size + 1 : i]
            window_slice = data_obj.create_df_obj(slice)
            predicted_value, cond_vol  = predict_with_fbm(window_slice, window_size, 1).values()
            
            dates.append(ts.index[i])
            predictions.append(predicted_value[0])
            cond_vols.append(cond_vol)

        # Convert predictions into DF and return it
        return pd.DataFrame({'predicted': predictions, 'conditional_vol': cond_vols}, index=pd.to_datetime(dates))


def rescale_columns(df: pd.DataFrame, columns: list, scale_factor: float) -> pd.DataFrame:
    """Rescales specified columns in a DF by scale factor."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col] / scale_factor
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    return df