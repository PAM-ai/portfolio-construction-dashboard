import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def get_factors_breakdown(review_data_output, review_date, factor, percentage=False):
    """
    Generate sector breakdown of a specific factor.
    
    Parameters:
        review_data_output (DataFrame): Combined review data
        review_date (datetime): Date for analysis
        factor (str): Factor to analyze
        percentage (bool): Whether to normalize as percentage
        
    Returns:
        tuple: (benchmark_factor_by_sector, index_factor_by_sector)
    """
    # Select data for the specified review date
    review_subset = review_data_output[review_data_output["Review Date"] == review_date]
    
    # Calculate weighted factor values by sector
    bmk_factor = review_subset.groupby("Sector").apply(lambda df: (df["Weight"] * df[factor]).sum())
    index_factor = review_subset.groupby("Sector").apply(lambda df: (df["IndexWeights"] * df[factor]).sum())
    
    # Convert to percentage if requested
    if percentage:
        bmk_factor = round(100 * (bmk_factor / bmk_factor.sum(axis=0)), 2)
        index_factor = round(100 * (index_factor / index_factor.sum(axis=0)), 2)

    return bmk_factor, index_factor

def plot_factors_breakdown(bmk_item, index_item, factor_name, sorting_column):
    """
    Generate a bar chart comparing factor breakdown between benchmark and index.
    
    Parameters:
        bmk_item (Series): Benchmark factor values by sector
        index_item (Series): Index factor values by sector
        factor_name (str): Name of the factor
        sorting_column (str): Column to sort by
        
    Returns:
        plotly.Figure: Comparison chart
    """
    # Convert to DataFrame for plotting
    df_plot = pd.DataFrame({
        "Sector": bmk_item.index,
        "Benchmark": bmk_item.values,
        "Index": index_item.values
    })

    # Sort values as requested
    df_plot = df_plot.sort_values(sorting_column, ascending=False)
    
    # Prepare data for grouped bar chart
    df_plot = df_plot.melt(id_vars="Sector", var_name="Portfolio", value_name="Value")

        # Define custom colour palette
    color_palette = {
        "Benchmark": "#4682B4",  # 
        "Index": "#228B22"       # 
    }

    # Create grouped bar chart
    factors_fig = px.bar(
        df_plot,
        x="Sector",
        y="Value",
        color="Portfolio",
        #color_discrete_map=color_palette,
        barmode="group",
        title=f"{factor_name} Breakdown by Sector",
        labels={"Value": f"{factor_name}", "Sector": "Sector"},
        template="plotly_white"
    )
    
    # Improve layout
    factors_fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return factors_fig

def display_targets_evolution(achieved_targets_df, selected_item):

        #Filter achieved targets DataFrame to compare review and achieved intensities
        achieved_targets_df = achieved_targets_df[[col for col in achieved_targets_df.columns if "ReviewIntensity" in col or "Achieved" in col or "Review Date" in col]]
        df_melt = achieved_targets_df.melt(id_vars="Review Date", var_name="Metric", value_name="Value") # Melt the DataFrame for easier plotting

        df_melt = df_melt[df_melt["Metric"].str.contains(selected_item)] # Filter for selected factor
        selected_item_mapping= {
             f"Achieved{selected_item}": "Index Intensity",
             f"ReviewIntensity_{selected_item}": "Benchmark Intensity"
        }
        df_melt["Metric"] = df_melt["Metric"].map(selected_item_mapping)
        fig_title = f"Evolution of {selected_item} Intensity Over Time" # Set title
        
        # Define custom colour palette
        color_palette = {
            "Benchmark Intensity": "#4682B4",  # 
            "Index Intensity": "#228B22"       # 
        }

        # Plotly line chart over time
        fig = px.line(
            df_melt, 
            x="Review Date", 
            y="Value", 
            color="Metric",
            #color_discrete_map=color_palette,
            title=fig_title,
            labels={"Value": "Factor Intensity", "Review Date": "Date"},
            template="plotly_white"
        )

        fig.update_layout(legend=dict(orientation="v", yanchor="bottom", y=1.02, xanchor="right", x=1)) # Change position of legend to vertical
        fig.update_traces(mode="lines+markers") # Add markes

        # Plot chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

def display_selection_menu(review_data_output, sustainable_factors):
        
        # Set up the widgets for user input
        col1, col2 = st.columns(2)
        with col1:
            selected_factor = st.selectbox(
                "Select Intensity Metric:", 
                options=sustainable_factors,
                key="factor_selector"
            )

        with col2:
            selected_date = st.selectbox(
                "Select Review Date:", 
                options=sorted(review_data_output["Review Date"].unique()),
                key="date_selector"
            )
        
        
        
        # Add options for percentage and sorting
        col3, col4 = st.columns(2)
        with col3:
            show_as_percentage = st.checkbox("Show as percentage of total", value=False)
        
        with col4:
            sort_by_bmk = st.checkbox("Sort by benchmark values", value=True)
        
        return selected_factor, selected_date, show_as_percentage, sort_by_bmk

def display_factor_breakdown(review_data_output, selected_factor, selected_date, show_as_percentage, sort_by_bmk):
        

        # Get the factor breakdown for the selected date and factor for each sector
        bmk_factor, index_factor = get_factors_breakdown(
            review_data_output, 
            selected_date, 
            selected_factor, 
            show_as_percentage
        )
        
        sorting_column = "Benchmark" if sort_by_bmk else "Index"

        # Create a bar chart comparing the benchmark and index factor breakdown
        factors_fig = plot_factors_breakdown(bmk_factor, index_factor, selected_factor, sorting_column)

        # Plot chart in Streamlit
        st.plotly_chart(factors_fig, use_container_width=True)