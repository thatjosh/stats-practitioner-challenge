import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t

def rolling_garch_price_forecast(
    df: pd.DataFrame,
    window: int = 250,
    dist: str = 'normal'
) -> pd.DataFrame:
    """Perform a rolling GARCH forecast to predict the next-day price.
    Uses a single random draw for each day to get a forecasted log-return
    and thus a forecasted price."""

    # Prepare columns for results
    df['predicted_price'] = np.nan
    df['forecasted_log_return'] = np.nan
    df['conditional_vol'] = np.nan
    
    for i in range(window, len(df)):
        # Fit a GARCH(1,1) on the past `window` returns
        rolling_slice = df.iloc[i-window:i]['log_returns']
        
        # Choose the distribution
        if dist == 'normal':
            am = arch_model(rolling_slice, mean='Zero', vol='GARCH', p=1, q=1, dist='normal', rescale=False)
        else:
            am = arch_model(rolling_slice, mean='Zero', vol='GARCH', p=1, q=1, dist='t', rescale=False)
        
        res = am.fit(disp='off')
        
        # Forecast next-day variance
        fcast_var = res.forecast(horizon=1).variance.iloc[-1, 0]
        fcast_std = np.sqrt(fcast_var)
        
        # Draw a shock
        if dist == 'normal':
            shock = np.random.randn()  # N(0,1)
        else:
            nu = res.params.get('nu', np.inf)
            # Student-t random deviate, scaled for stdev=1
            shock = t.rvs(df=nu) / np.sqrt(nu / (nu - 2))
        
        # Next-day log-return (from the GARCH volatility * random shock)
        forecasted_log_return = fcast_std * shock
        
        # Price_{t+1} = Price_t * exp( forecasted_log_return )
        # Last observed price is at index i-1
        last_price = df.iloc[i-1]['price']
        predicted_price = last_price * np.exp(forecasted_log_return)
        
        # Store results
        df.iloc[i, df.columns.get_loc('forecasted_log_return')] = forecasted_log_return
        df.iloc[i, df.columns.get_loc('predicted_price')] = predicted_price
        df.iloc[i, df.columns.get_loc('conditional_vol')] = fcast_std

    return df