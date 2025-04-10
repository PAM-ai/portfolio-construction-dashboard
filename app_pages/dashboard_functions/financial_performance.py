import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def calculate_index_performance(review_data, prices, index_weights):
    """
    Calculate Index and Benchmark performance metrics.

    Parameters:
        review_data (DataFrame): Data with benchmark weights and close prices
        prices (DataFrame): Price history data
        index_weights (DataFrame): Index weights after constraints and exclusions

    Returns:
        tuple: (combined_data, benchmark_returns, index_returns)
    """
    # Convert dates to datetime for consistency
    review_data["Review Date"] = pd.to_datetime(review_data["Review Date"])
    index_weights["Review Date"] = pd.to_datetime(index_weights["Review Date"])

    # Merge index weights with review data
    review_data_output = review_data.merge(
        index_weights[["Review Date", "Symbol", "IndexWeights"]], 
        on=["Review Date", "Symbol"], 
        how="left"
    )
    review_data_output["IndexWeights"] = review_data_output["IndexWeights"].fillna(0)

    # Merge with prices data
    prices["Review Date"] = pd.to_datetime(prices["Review Date"])
    prices = prices.merge(
        review_data_output[["Review Date", "Symbol", "IndexWeights"]], 
        on=["Review Date", "Symbol"], 
        how="left"
    )

    # Compute daily stock returns
    prices['Return'] = prices.sort_values("Date", ascending=True).groupby('Symbol')['Close'].pct_change()
    prices = prices.dropna()  # Drop NaN returns (first date)

    # Compute weighted returns
    prices['Weighted Benchmark Return'] = prices['Return'] * prices['Weight']
    prices['Weighted Index Return'] = prices['Return'] * prices['IndexWeights']

    # Aggregate to get total return per date
    benchmark_returns = prices.groupby('Date')['Weighted Benchmark Return'].sum()
    index_returns = prices.groupby('Date')['Weighted Index Return'].sum()

    # Verify weight sums (debugging)
    benchmark_weight_sums = prices.groupby("Date")["Weight"].sum().sort_values()
    index_weight_sums = prices.groupby("Date")["IndexWeights"].sum().sort_values()
    
    if not (0.99 <= benchmark_weight_sums.mean() <= 1.01) or not (0.99 <= index_weight_sums.mean() <= 1.01):
        st.warning(f"⚠️ Weight sums may not equal 100%. Check your data.")
    
    # Change date format for future displays
    review_data_output["Review Date"] = pd.to_datetime(review_data_output["Review Date"]).dt.date

    return review_data_output, benchmark_returns, index_returns

def calculate_metrics(portfolio_returns, benchmark_returns):
    """Calculate key risk metrics"""
    # Annualization factor (assuming daily returns)
    ann_factor = 252
    
    # Active returns
    active_returns = portfolio_returns - benchmark_returns
    
    # Tracking Error
    tracking_error = np.std(active_returns) * np.sqrt(ann_factor)
    
    # Information Ratio
    information_ratio = np.mean(active_returns) * ann_factor / tracking_error
    
    # Volatility
    portfolio_vol = np.std(portfolio_returns) * np.sqrt(ann_factor)
    benchmark_vol = np.std(benchmark_returns) * np.sqrt(ann_factor)
    
    # Sharpe Ratio (assuming risk-free rate of 2%)
    rf_rate = 0.02 / ann_factor
    portfolio_sharpe = (np.mean(portfolio_returns) - rf_rate) * ann_factor / portfolio_vol
    benchmark_sharpe = (np.mean(benchmark_returns) - rf_rate) * ann_factor / benchmark_vol
    
    # Max Drawdown
    portfolio_cum_returns = (1 + portfolio_returns).cumprod()
    benchmark_cum_returns = (1 + benchmark_returns).cumprod()
    
    portfolio_drawdown = 1 - portfolio_cum_returns / portfolio_cum_returns.cummax()
    benchmark_drawdown = 1 - benchmark_cum_returns / benchmark_cum_returns.cummax()
    
    portfolio_max_drawdown = portfolio_drawdown.max()
    benchmark_max_drawdown = benchmark_drawdown.max()
    
    return {
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'portfolio_vol': portfolio_vol,
        'benchmark_vol': benchmark_vol,
        'portfolio_sharpe': portfolio_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'portfolio_max_drawdown': portfolio_max_drawdown,
        'benchmark_max_drawdown': benchmark_max_drawdown,
        'drawdown_series': {
            'portfolio': portfolio_drawdown,
            'benchmark': benchmark_drawdown
        }
    }

def risk_metrics_dashboard(portfolio_returns, benchmark_returns, dates):
    """Display the risk metrics dashboard"""
    metrics = calculate_metrics(portfolio_returns, benchmark_returns)
    
    # Create columns for KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Tracking Error (%)",
            value=f"{metrics['tracking_error']*100:.2f}%"
        )
        st.metric(
            label="Information Ratio",
            value=f"{metrics['information_ratio']:.2f}"
        )
    
    with col2:
        st.metric(
            label="Portfolio Volatility",
            value=f"{metrics['portfolio_vol']*100:.2f}%",
            delta=f"{(metrics['portfolio_vol'] - metrics['benchmark_vol'])*100:.2f}%",
            delta_color="inverse"
        )
        st.metric(
            label="Benchmark Volatility",
            value=f"{metrics['benchmark_vol']*100:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Portfolio Sharpe",
            value=f"{metrics['portfolio_sharpe']:.2f}",
            delta=f"{metrics['portfolio_sharpe'] - metrics['benchmark_sharpe']:.2f}",
            delta_color="normal"
        )
        st.metric(
            label="Benchmark Sharpe",
            value=f"{metrics['benchmark_sharpe']:.2f}"
        )
    
    with col4:
        st.metric(
            label="Max Drawdown (Portfolio)",
            value=f"{metrics['portfolio_max_drawdown']*100:.2f}%",
            delta=f"{(metrics['benchmark_max_drawdown'] - metrics['portfolio_max_drawdown'])*100:.2f}%",
            delta_color="normal"
        )
        st.metric(
            label="Max Drawdown (Benchmark)",
            value=f"{metrics['benchmark_max_drawdown']*100:.2f}%"
        )

def plot_index_performance(benchmark_returns, index_returns):
    """
    Generate a comparative performance chart.
    
    Parameters:
        benchmark_returns (Series): Benchmark daily returns
        index_returns (Series): Index daily returns
        
    Returns:
        plotly.Figure: Performance comparison chart
    """
    benchmark_performance = (1 + benchmark_returns).cumprod()
    index_performance = (1 + index_returns).cumprod()

    # Create DataFrame for plotting
    perf_df = pd.DataFrame({
        "Date": benchmark_performance.index,
        "Benchmark": benchmark_performance.values,
        "Index": index_performance.values
    })

    # Melt DataFrame for Plotly format
    perf_df = perf_df.melt(id_vars="Date", var_name="Portfolio", value_name="Cumulative Return")

    # Define custom colour palette
    color_palette = {
        "Benchmark": "#EF553B",  # 
        "Index": "#636EFA"       # 
    }

    # Plot using Plotly Express
    fig = px.line(
        perf_df, 
        x="Date", 
        y="Cumulative Return", 
        color="Portfolio",
        #color_discrete_map=color_palette, 
        title="Cumulative Performance: Index vs. Benchmark",
        labels={"Cumulative Return": "Growth of $1 Invested", "Date": "Date"},
        template="plotly_white"
    )
    
    # Enhance plot appearance
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )

    return fig
