import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import portfolio_construction as ptf
from importlib import reload

def calculate_index_performance(review_data, prices, index_weights):
    """
    Calculate Index and Benchmark performance

    Parameters:
    review_data: Review data with benchmark weights and close prices
    index_weights: Index weights after meeting constraints and exclusions

    Returns:
    benchmark_returns: Returns of benchmark
    index_returns: Returns of index
    """

    # Merge with original data to get weights even for excluded companies
    review_data["Review Date"] = pd.to_datetime(review_data["Review Date"])
    index_weights["Review Date"] = pd.to_datetime(index_weights["Review Date"])

    review_data_output = review_data.merge(index_weights[["Review Date", "Symbol", "IndexWeights"]], on=["Review Date", "Symbol"], how="left")
    review_data_output["IndexWeights"] = review_data_output["IndexWeights"].fillna(0)

    # Merge with prices data
    prices["Review Date"] = pd.to_datetime(prices["Review Date"])
    prices = prices.merge(review_data_output[["Review Date", "Symbol", "IndexWeights"]], on=["Review Date", "Symbol"], how="left")

    # Compute daily stock returns
    prices['Return'] = prices.sort_values("Date", ascending=True).groupby('Symbol')['Close'].pct_change()

    # Drop NaN returns (first date)
    prices = prices.dropna()

    # Compute Index Return as a weighted sum
    prices['Weighted Benchmark Return'] = prices['Return'] * prices['Weight']
    prices['Weighted Index Return'] = prices['Return'] * prices['IndexWeights']

    # Aggregate to get total index return per date
    benchmark_returns = prices.groupby('Date')['Weighted Benchmark Return'].sum()
    index_returns = prices.groupby('Date')['Weighted Index Return'].sum()

    print(prices.groupby("Date")["Weight"].sum().sort_values(), prices.groupby("Date")["IndexWeights"].sum().sort_values())
    return review_data_output, benchmark_returns, index_returns

def calculate_financial_metrics(returns, annualization_factor):
    """Calculate and return financial metrics for given returns."""
    
    # Calculate cumulative returns
    cumulative_return = (1 + returns).cumprod() - 1
    
    # Calculate Sharpe Ratio (assuming a risk-free rate of 0)
    annualized_return = returns.mean() * annualization_factor  # assuming daily returns
    annualized_volatility = returns.std() * np.sqrt(annualization_factor)  # assuming daily returns
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    
    # Calculate Max Drawdown
    cumulative_max = cumulative_return.cummax()
    drawdown = cumulative_return - cumulative_max
    max_drawdown = drawdown.min()    
    
    metrics = {
        'Cumulative Return': cumulative_return[-1],
        'Annualized Return': annualized_return,
        'Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }
    
    return metrics

def plot_index_performance(benchmark_returns, index_returns):
    benchmark_performance = (1 + benchmark_returns).cumprod()
    index_performance = (1 + index_returns).cumprod()

    # Create DataFrame for Plotly
    perf_df = pd.DataFrame({
        "Review Date": benchmark_performance.index,
        "Benchmark": benchmark_performance.values,
        "Index": index_performance.values
    })

    # Melt DataFrame for Plotly format
    perf_df = perf_df.melt(id_vars="Review Date", var_name="Portfolio", value_name="Cumulative Return")

    # Plot using Plotly Express
    fig = px.line(
        perf_df, 
        x="Review Date", 
        y="Cumulative Return", 
        color="Portfolio", 
        title="Cumulative Performance: Index vs. Benchmark",
        labels={"Cumulative Return": "Cumulative Return (%)", "Review Date": "Date"},
        template="plotly_dark"
    )

    return fig

def get_factors_reduction(review_data_output, review_date, selected_item, percentage):
    
    # Select subset
    review_subset = review_data_output[review_data_output["Review Date"]==review_date]
    
    # Group by sector
    bmk_item = review_subset.groupby("Sector").apply(lambda df: (df["Weight"] * df[selected_item]).sum())
    index_item = review_subset.groupby("Sector").apply(lambda df: (df["IndexWeights"] * df[selected_item]).sum())
    
    if percentage:
        bmk_item = round(100 * (bmk_item / bmk_item.sum(axis=0)), 2)
        index_item = round(100 * (index_item / index_item.sum(axis=0)), 2)

    return bmk_item, index_item

def plot_factors_reduction(bmk_item, index_item, selected_item, sorting_column):

    # Convert to DataFrame for Plotly
    df_plot = pd.DataFrame({
        "Sector": bmk_item.index,
        "Benchmark": bmk_item.values,
        "Index": index_item.values
    })

    df_plot = df_plot.sort_values(sorting_column, ascending=False)

    df_plot = df_plot.melt(id_vars="Sector", var_name="Portfolio", value_name="Weighted Value")

    # Create grouped histogram
    factors_fig = px.bar(
        df_plot,
        x="Sector",
        y="Weighted Value",
        color="Portfolio",
        barmode="group",
        title=f"Comparison of {selected_item} Between Benchmark and Index",
        labels={"Weighted Value": f"Weighted {selected_item}", "Sector": "Sector"}
        )

    return factors_fig

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from importlib import reload

# Function to switch views
def switch_view():
    st.session_state["current_view"] = "Sustainable" if st.session_state["current_view"] == "Financial" else "Financial"

# Initialize session state for the view
if "current_view" not in st.session_state:
    st.session_state["current_view"] = "Financial"

def index_generation_page():
    """Page for running the optimization and generating the index."""

    # Reload proprietary package
    reload(ptf)

    # Ensure data exists before proceeding
    if "review_data" not in st.session_state or "prices" not in st.session_state:
        st.error("⚠️ Please configure portfolio in Step 1 first.")
        st.stop()

    review_data = st.session_state["review_data"]
    prices = st.session_state["prices"]

    st.header("Step 2: Generate Optimized Index")


    # Generate index weights
    constraints = st.session_state.get("constraints", {})
    if not constraints:
        st.warning("No constraints set. Using default values.")

    review_dates = list(review_data["Review Date"].unique())
    sustainable_factors = list(constraints.keys())
    excluded = st.session_state["excluded_sub_sectors"]
    config = st.session_state["config"]
    target_levels = np.array([v["target_reduction"] for v in constraints.values()])
    trajectory_rates = np.array([v["annual_reduction_rate"] for v in constraints.values()])
    
    targets_df = ptf.get_targets(review_data, sustainable_factors, target_levels, trajectory_rates, reviews_per_year=2)
    index_weights, achieved_targets_df = ptf.get_weights(
        review_dates, review_data, sustainable_factors, excluded, targets_df, "exp", config
    )

    # Calculate performance
    review_data_output, benchmark_returns, index_returns = calculate_index_performance(review_data, prices, index_weights)

    # Toggle button (top-left)
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        other_view = "Sustainable" if st.session_state["current_view"] == "Financial" else "Financial"

        # Apply custom styling to increase button width
        st.markdown(
            """
            <style>
                div.stButton > button {
                    width: 100%;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Create the button
        st.button(f"🔄 Switch View to {other_view} Panel", on_click=switch_view)

    # Determine which section to show
    if st.session_state["current_view"] == "Financial":
        st.subheader("📈 Financial Analysis")

        # Plot performance comparison
        index_perf_fig = plot_index_performance(benchmark_returns, index_returns)
        st.plotly_chart(index_perf_fig, use_container_width=True)

        # Display financial metrics
        st.write("### Financial Metrics for Benchmark")
        bmk_metrics = calculate_financial_metrics(benchmark_returns, 252)
        st.json(bmk_metrics)

        st.write("### Financial Metrics for Index")
        index_metrics = calculate_financial_metrics(index_returns, 252)
        st.json(index_metrics)

        st.write(review_data_output)

    else:
        st.subheader("🌱 Sustainable Analysis")

        # Extract constraints        
        st.write(review_data_output)

        with st.form(key="trajectory_form"):
            selected_item_trajectory = st.selectbox("Select Item for trajectory analysis", sustainable_factors)
            submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            
            df_melt = achieved_targets_df.melt(id_vars="Review Date", var_name="Metric", value_name="Value")
            df_melt = df_melt[df_melt["Metric"].str.contains(selected_item_trajectory)]
            fig_title = f"Evolution of {selected_item_trajectory} Intensity Over Time"
        
        else:
            
            df_melt = achieved_targets_df.melt(id_vars="Review Date", var_name="Metric", value_name="Value")
            fig_title = "Evolution of Targets Intensities Over Time"

        fig = px.line(
            df_melt, 
            x="Review Date", 
            y="Value", 
            color="Metric", 
            title=fig_title,
            labels={"Value": "Factor Intensity"}
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.form("selection_form"):
            selected_date = st.selectbox("Select review date", list(review_data["Review Date"].unique()))
            selected_item = st.selectbox("Select item", sustainable_factors)
            show_as_percentage = st.checkbox("Show as percentage of total", value=False)
            sort_by_bmk = st.checkbox("Sort by benchmark values", value=True)
            submit_button = st.form_submit_button("Submit")

        if submit_button:
            bmk_item, index_item = get_factors_reduction(review_data_output, selected_date, selected_item, show_as_percentage)
            sorting_column = "Benchmark" if sort_by_bmk else "Index"
            factors_fig = plot_factors_reduction(bmk_item, index_item, selected_item, sorting_column)
            st.plotly_chart(factors_fig, use_container_width=True)
