import os
import streamlit as st
import pandas as pd
import numpy as np
import app_pages.dashboard_functions.utility_functions as ut
import app_pages.dashboard_functions.financial_performance as perf
import app_pages.dashboard_functions.carbon_footprint as footprint
import app_pages.dashboard_functions.carbon_attribution as attribution
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import app_pages.portfolio.target_portfolio as ptf
import io
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from importlib import reload

reload(ut)
reload(perf)
reload(footprint)
reload(attribution)
reload(ptf)

# ---- APP LAYOUT AND FUNCTIONALITY ----

def index_generation_page():
    """Page for running the tilting resolution and generating the index."""
    # Add page styling
    st.markdown("""
    <style>
        .block-container {max-width: 1600px}
        .stButton > button {width: 100%}
        div[data-testid="stMetricValue"] {font-size: 1.5rem}
        .review-data {height: 300px; overflow-y: scroll}
    </style>
    """, unsafe_allow_html=True)

    # Reload proprietary package for fresh data
    
    st.title("Sustainable Index Construction Dashboard - Generate Index")
    # Ensure data exists before proceeding
    if "review_data" not in st.session_state or "prices" not in st.session_state:
        st.error("⚠️ Please upload and configure portfolio data in Step 1 first.")
        st.stop()

    review_data = st.session_state["review_data"]
    prices = st.session_state["prices"]


    # Check for constraints
    constraints = st.session_state.get("constraints", {})
    if not constraints:
        st.warning("⚠️ No constraints set. Please add a constraint and choose your settings before navigating to this page.")
        st.stop()

    # Gather configuration parameters
    review_dates = sorted(review_data["Review Date"].unique())
    sustainable_factors = list(constraints.keys())
    excluded = st.session_state.get("excluded_sub_sectors", [])
    config = st.session_state.get("config", {})
    
    # Extract target levels and trajectory rates
    target_levels = np.array([v["target_reduction"] for v in constraints.values()])
    trajectory_rates = np.array([v["annual_reduction_rate"] for v in constraints.values()])
    
    with st.spinner("Generating index weights..."):
        # Get targets trajectory

        targets_df = ptf.get_targets(
            review_data, 
            sustainable_factors, 
            target_levels, 
            trajectory_rates, 
            reviews_per_year=2
        )
        
        # Get weights that satisfy the constraints and factors exposures
        index_weights, achieved_targets_df = ptf.get_weights(
            review_dates, 
            review_data, 
            sustainable_factors, 
            excluded, 
            targets_df, 
            "exp", 
            config
        )

    # Calculate performance metrics
    with st.spinner("Calculating performance..."):
        review_data_output, benchmark_returns, index_returns = perf.calculate_index_performance(
            review_data, 
            prices, 
            index_weights
        )

    with st.expander("🧩 **Don't miss anything. Click here to learn how to use this page**", expanded=False):
        st.markdown("""
        Welcome! Here’s a quick guide to help you navigate and get the most out of this dashboard — **click again to collapse this section**:

        1. **Navigation between menus:**
            - **Financial Menu**: Analyze the financial performance of your portfolio.
            - **Sustainable Menu**: Explore the sustainability profile of your portfolio.

        2. **Exporting your results:**
            - Once you're satisfied with your analysis, **export the optimized portfolio weights to Excel** using the export button in the sidebar.

        3. **Adjusting portfolio settings:**
            - To modify portfolio constraints or preferences, **click the button below to return to the configuration page**.

        Enjoy your analysis! 🚀
        """)



    # Get constraints in dataframe format for display and export
    constraints_df = pd.DataFrame({
    "Constraint": sustainable_factors,
    "Reduction": [round(-100 * value) for value in target_levels],
    "Trajectory": [round(-100 * value) for value in trajectory_rates]
    })
    constraints_df.set_index('Constraint', inplace=True)

    weights_constraints_df = pd.DataFrame(list(config.items()), columns=["Constraint", "Value"])
    weights_constraints_df = weights_constraints_df[weights_constraints_df["Constraint"]!="Stock Bound"]
    weights_constraints_df.set_index("Constraint", inplace=True)

    # Export button
    st.sidebar.subheader("Satisfied with your results ?")
    ut.create_excel_download_button(index_weights, constraints_df.reset_index(), weights_constraints_df.reset_index(), pd.DataFrame(excluded))

    # Recap constraints
    st.sidebar.subheader("Settings Recap")

    # Display in sidebar as a table
    st.sidebar.table(constraints_df)
    st.sidebar.table(weights_constraints_df)

    st.sidebar.subheader("SubSectors excluded")
    # Create a Markdown string for bullet points
    excluded_bullet_points = '\n'.join([f"- {item}" for item in excluded])
    if excluded_bullet_points:
        st.sidebar.markdown(excluded_bullet_points) # Display in Streamlit
    else:
        st.sidebar.write("No Subsectors excluded for this index")

    if st.button("⬅️ **Press here to Change Portfolio Targets and Settings** ", type="primary", use_container_width=True):
      ut.go_to("Select Constraints")

    # Create tabs for Financial and Sustainable Analysis
    tab1, tab2 = st.tabs(["📈 **Financial Menu**", "🌱 **Sustainable Menu**"])

    # Financial Analysis Tab
    with tab1:
        st.markdown("> 👋 **Tip:** Don't miss the 🌱 Sustainable Menu to explore your portfolio's environmental impact!")

        # Risk Metrics Dashboard 
        st.subheader("Performance Metrics")
        perf.risk_metrics_dashboard(index_returns, benchmark_returns, prices["Date"])

        # Portfolio Performance
        st.subheader("Portfolio Performance")
        index_perf_fig = perf.plot_index_performance(benchmark_returns, index_returns)
        st.plotly_chart(index_perf_fig, use_container_width=True)

       

    # Sustainable Analysis Tab
    with tab2:
        
        st.markdown("> 📊 **Reminder:** Check the 📈 Financial Menu for performance and risk metrics.")

        # User input menu for sustainable factors
        st.markdown("#### Select Factor for Analysis")
        selected_factor, selected_date, show_as_percentage, sort_by_bmk = footprint.display_selection_menu(review_data_output, sustainable_factors)
        # SECTION 1: Factor Breakdown Analysis
        # Calculate and display factor breakdown
        st.markdown("#### Sector Breakdown Analysis")
        footprint.display_factor_breakdown(review_data_output, selected_factor, selected_date, show_as_percentage, sort_by_bmk)

        # SECTION 2: Intensity Decomposition Analysis over Time
        st.markdown("#### Intensity Reduction Analysis (Brinson Style)")
        attribution.brinson_attribution_dashboard(review_data_output, achieved_targets_df, selected_factor, selected_date)
        
        


