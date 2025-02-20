import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2

def compute_hit_ratio(df, hurst_col='hurst', return_col='log_returns', threshold=0.5):
    """Compute the hit ratio for a trading strategy based on the Hurst exponent."""
    df = df.dropna().copy()  # Ensure no NaNs

    # Generate trading signals based on Hurst exponent
    df['signal'] = np.where(df[hurst_col] > threshold, 1, -1)  # 1 for trending, -1 for mean-reverting

    # Shift returns forward to compare with the signal
    df['future_return'] = df[return_col].shift(-1)  # Next-period return

    # Correct predictions: signal * future return > 0 means correct prediction
    df['correct'] = np.sign(df['signal'] * df['future_return']) == 1

    # Compute hit ratio
    hit_ratio = df['correct'].mean()
    return hit_ratio

def perform_jarque_bera_test(residuals):
    """Perform the Jarque-Bera test to check for normality in residuals."""
    jb_stat, p_value = stats.jarque_bera(residuals)
    skewness = residuals.skew()
    kurtosis = residuals.kurtosis()

    return {
        'JB Statistic': jb_stat,
        'p-value': p_value,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Normality': 'Reject H0 (Not Normal)' if p_value < 0.05 else 'Fail to Reject H0 (Normal)'
    }

def compute_var_violations(df, var_col, return_col, confidence_level=0.99):
    df = df.dropna().copy()
    
    # Identify VaR exceedances (violations)
    df['exceedance'] = df[return_col] < df[var_col]  # True if actual return is less than VaR

    # Count actual exceedances
    actual_exceedances = df['exceedance'].sum()

    # Expected exceedances based on confidence level
    expected_exceedances = len(df) * (1 - confidence_level)

    # Violation ratio
    violation_ratio = actual_exceedances / expected_exceedances if expected_exceedances > 0 else np.nan

    return {
        'actual_exceedances': actual_exceedances,
        'expected_exceedances': expected_exceedances,
        'violation_ratio': violation_ratio
    }

def bernoulli_coverage_test(df, var_col, return_col, confidence_level=0.99):
    df = df.dropna().copy()
    p = 1 - confidence_level
    N = len(df)
    
    # Identify VaR exceedances (violations)
    df['exceedance'] = df[return_col] < df[var_col]  # True if actual return is less than VaR

    # Count actual exceedances
    X = df['exceedance'].sum()

    # Expected exceedances based on confidence level
    p_hat = X / N

    # Compute ikelihood ratio test statistic
    if p_hat == 0 or p_hat == 1:  # Avoid log(0) errors
        LR_POF = 0
    else:
        LR_POF = -2 * (np.log((1 - p) ** (N - X) * p**X) - np.log((1 - p_hat) ** (N - X) * p_hat**X))

    p_value = 1 - chi2.cdf(LR_POF, df=1)

    return p_value, LR_POF