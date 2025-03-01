import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t
from enum import Enum
class Distribution(Enum):
    NORMAL = 'normal'
    T = 't'

def rolling_garch_price_forecast(
    df: pd.DataFrame,
    window: int = 250,
    dist: Distribution = Distribution.NORMAL,
) -> pd.DataFrame:
    """Perform a rolling GARCH forecast to predict the next-day price and compute log returns."""

    # Initialise columns for results
    df['predicted_price'] = np.nan
    df['predicted_log_return'] = np.nan
    df['conditional_vol'] = np.nan
    
    for i in range(window, len(df)):
        # Fit a GARCH(1,1) on the past `window` returns
        rolling_slice = df.iloc[i-window:i]['log_returns']
        
        # Choose the distribution
        if dist == Distribution.NORMAL:
            am = arch_model(rolling_slice, mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
        elif dist == Distribution.T:
            am = arch_model(rolling_slice, mean='Zero', vol='GARCH', p=1, q=1, dist='t')
        
        res = am.fit(disp='off')
        
        # Forecast next-day variance
        fcast_var = res.forecast(horizon=1).variance.iloc[-1, 0] 
        fcast_std = np.sqrt(fcast_var) / 100
        
        # Draw a shock
        if dist == Distribution.NORMAL:
            shock = np.random.randn()  # N(0,1)
        elif dist == Distribution.T:
            nu = res.params.get('nu', np.inf)
            # Student-t random deviate, scaled for stdev=1
            shock = t.rvs(df=nu) / np.sqrt(nu / (nu - 2))
        
        # Next-day log-return (from the GARCH volatility * random shock)
        predicted_log_return = fcast_std * shock
        
        # Price_{t+1} = Price_t * exp( predicted_log_return )
        # Last observed price is at index i-1
        last_price = df.iloc[i-1]['price']
        predicted_price = last_price * np.exp(predicted_log_return)
        
        # Store results
        df.iloc[i, df.columns.get_loc('predicted_log_return')] = predicted_log_return
        df.iloc[i, df.columns.get_loc('predicted_price')] = predicted_price
        df.iloc[i, df.columns.get_loc('conditional_vol')] = fcast_std
    return df