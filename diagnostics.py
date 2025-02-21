import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox

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

def compute_var_violations(df, var_col, predicted_col, confidence_level=0.99):
    df = df.dropna().copy()
    
    # Identify VaR exceedances (violations)
    df['exceedance'] = df[predicted_col] < df[var_col]  # True if actual return is less than VaR

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

def bernoulli_coverage_test(df, var_col, predicted_col, confidence_level=0.99):
    
    df = df.dropna().copy()
    p = 1 - confidence_level
    N = len(df)
    
    # Identify VaR exceedances (violations)
    df['exceedance'] = df[predicted_col] < df[var_col]  # True if actual return is less than VaR

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

def in_sample_diagnostics(actual_log_returns: pd.Series, predicted_log_returns: pd.Series, cond_volatility: pd.Series):
    """
    Compute diagnostics for a time series forecasting model.
    
    Performs:
    1) Jarque-Bera test for normality on residuals.
    2) Ljung-Box test on residuals and squared residuals to check for autocorrelation.
    """

    # Compute residuals (forecast error)
    residuals = actual_log_returns - predicted_log_returns
    residuals = residuals.dropna()  # Ensure no NaNs
    cond_volatility = cond_volatility.dropna()
    std_residuals =  residuals / cond_volatility

    # 1) Jarque-Bera test for normality
    _, jb_pvalue = jarque_bera(std_residuals)
    print(f"Jarque-Bera test p-value: {jb_pvalue:.5f}")

    # 2) Ljung-Box test on residuals and squared residuals (first 10 lags)
    lb1 = acorr_ljungbox(std_residuals, lags=[10], return_df=False)
    lb2 = acorr_ljungbox(std_residuals**2, lags=[10], return_df=False)
    
    print(f"Ljung-Box (residuals) p-value, {lb1[1][0]:.5f}")
    print(f"Ljung-Box (residuals^2) p-value, {lb2[1][0]:.5f}")


def compute_hit_rate(predicted: pd.Series, actual: pd.Series) -> float:
    """Compute the hit rate for forecasted returns versus actual log returns."""
    sign_match = np.sign(predicted) == np.sign(actual)
    hit_rate = sign_match.mean()
    print(f"\nHit Rate: {hit_rate * 100:.2f}%")
    return hit_rate