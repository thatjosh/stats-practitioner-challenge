import plotly.graph_objects as go
import pandas as pd

def actual_vs_predicted_time_series_plot(
        df: pd.DataFrame, actual_col_name: str, predicted_col_name: str, var_name: str
    ):
    """Plots the actual vs predicted values for a variable from the provided DataFrame."""
    
    fig = go.Figure()
    
    # Add trace for predicted values
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[predicted_col_name],
        mode='lines',
        name='Predicted',
        line=dict(color='blue')
    ))
    
    # Add trace for actual values
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[actual_col_name],
        mode='lines',
        name='Actual',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=f"Predicted vs Actual {var_name}",
        xaxis_title="Date",
        yaxis_title=var_name,
        template="plotly_white"
    )
    
    fig.show()

def plot_var_violations(df, var_col, predicted_col, title="VaR Violations"):
    df = df.dropna().copy()

    # Identify VaR violations
    df['exceedance'] = df[predicted_col] < df[var_col]

    fig = go.Figure()

    # Add trace for actual values
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df[predicted_col], 
        mode='lines',
        name="Actual Returns", 
        line=dict(color='blue'), 
        opacity=0.6
    ))

    # Plot VaR threshold
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df[var_col], 
        mode='lines', 
        name="VaR Threshold", 
        line=dict(color='orange', dash='dot')
    ))

    # Plot VaR violations as red markers
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Log Returns",
        legend=dict(x=0, y=1),
        hovermode="x unified",
        template="plotly_white"
    )

    fig.show()