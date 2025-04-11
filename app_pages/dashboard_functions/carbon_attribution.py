import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import app_pages.dashboard_functions.carbon_footprint as footprint
from importlib import reload

reload(footprint)

def perform_brinson_attribution(review_data_output, review_date, intensity_metric, sector_column='Sector'):
    """
    Performs Brinson-style attribution analysis on intensity metrics between portfolio and benchmark
    
    Parameters:
    -----------
    review_data_output : pandas.DataFrame
        DataFrame containing both portfolio and benchmark data
    
    review_date : datetime or str
        The review date for analysis
    
    intensity_metric : str
        The intensity metric to analyze (column name in the DataFrame)
    
    sector_column : str, default='Sector'
        The column name containing sector classifications
        
    Returns:
    --------
    dict
        Dictionary containing attribution results
    """

    # Filter data for the review date
    data = review_data_output[review_data_output["Review Date"] == review_date].copy()
    
    # Ensure weights sum to 1 for both portfolio and benchmark
    data['Weight'] = data['Weight'] / data['Weight'].sum()
    data['IndexWeights'] = data['IndexWeights'] / data['IndexWeights'].sum()
    
    # Group by sector and calculate sector weights and intensities
    sectors = data.groupby(sector_column).apply(
        lambda x: pd.Series({
            'portfolio_weight': x['IndexWeights'].sum(),
            'portfolio_intensity': (x[intensity_metric] * x['IndexWeights']).sum() / x['IndexWeights'].sum() if x['IndexWeights'].sum() > 0 else 0,
            'benchmark_weight': x['Weight'].sum(),
            'benchmark_intensity': (x[intensity_metric] * x['Weight']).sum() / x['Weight'].sum() if x['Weight'].sum() > 0 else 0
        })
    ).reset_index()
    
    # Calculate attribution effects for each sector
    sectors['allocation_effect'] = (sectors['portfolio_weight'] - sectors['benchmark_weight']) * sectors['benchmark_intensity']
    sectors['selection_effect'] = sectors['benchmark_weight'] * (sectors['portfolio_intensity'] - sectors['benchmark_intensity'])
    sectors['interaction_effect'] = (sectors['portfolio_weight'] - sectors['benchmark_weight']) * (sectors['portfolio_intensity'] - sectors['benchmark_intensity'])
    sectors['total_effect'] = sectors['allocation_effect'] + sectors['selection_effect'] + sectors['interaction_effect']
    
    # Calculate overall intensity for portfolio and benchmark
    portfolio_intensity = (sectors['portfolio_weight'] * sectors['portfolio_intensity']).sum()
    benchmark_intensity = (sectors['benchmark_weight'] * sectors['benchmark_intensity']).sum()
    intensity_difference = portfolio_intensity - benchmark_intensity
    
    # Aggregate effects
    total_allocation_effect = sectors['allocation_effect'].sum()
    total_selection_effect = sectors['selection_effect'].sum()
    total_interaction_effect = sectors['interaction_effect'].sum()
    
    # Calculate percent contribution of each effect (avoiding division by zero)
    if abs(intensity_difference) > 1e-10:
        allocation_pct = (total_allocation_effect / intensity_difference) * 100
        selection_pct = (total_selection_effect / intensity_difference) * 100
        interaction_pct = (total_interaction_effect / intensity_difference) * 100
    else:
        allocation_pct = selection_pct = interaction_pct = 0
    
    # Prepare return data
    attribution_results = {
        'review_date': review_date,
        'portfolio_intensity': portfolio_intensity,
        'benchmark_intensity': benchmark_intensity,
        'intensity_difference': intensity_difference,
        'allocation_effect': total_allocation_effect,
        'selection_effect': total_selection_effect,
        'interaction_effect': total_interaction_effect,
        'allocation_pct': allocation_pct,
        'selection_pct': selection_pct,
        'interaction_pct': interaction_pct,
        'sector_data': sectors
    }
    
    return attribution_results

def plot_attribution_waterfall(attribution_results, intensity_metric_name):
    """
    Create a waterfall chart showing Brinson attribution components
    
    Parameters:
    -----------
    attribution_results : dict
        Results from perform_brinson_attribution
    
    intensity_metric_name : str
        Name of the intensity metric for display
    
    Returns:
    --------
    plotly.Figure
    """
    # For intensity metrics, lower is better (green for negative values)
    fig = go.Figure(go.Waterfall(
        name = "Attribution",
        orientation = "v",
        measure = ["absolute", "relative", "relative", "relative", "total"],
        x = ["Benchmark", "Allocation", "Selection", "Interaction", "Portfolio"],
        textposition = "outside",
        text = [
            f"{attribution_results['benchmark_intensity']:.2f}",
            f"{attribution_results['allocation_effect']:.2f} ({attribution_results['allocation_pct']:.1f}%)",
            f"{attribution_results['selection_effect']:.2f} ({attribution_results['selection_pct']:.1f}%)",
            f"{attribution_results['interaction_effect']:.2f} ({attribution_results['interaction_pct']:.1f}%)",
            f"{attribution_results['portfolio_intensity']:.2f}"
        ],
        y = [
            attribution_results['benchmark_intensity'],
            attribution_results['allocation_effect'],
            attribution_results['selection_effect'],
            attribution_results['interaction_effect'],
            0  # The total will be calculated automatically
        ],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"green"}},  # Green for negative values (improvement)
        increasing = {"marker":{"color":"red"}},    # Red for positive values (worse)
        totals = {"marker":{"color":"#636EFA"}}   # Blue for totals
    ))
    
    review_date_text = f" for {attribution_results['review_date'].strftime('%Y-%m-%d')}"
    
    fig.update_layout(
        title = f"{intensity_metric_name} Intensity Attribution Analysis{review_date_text}",
        showlegend = False,
        xaxis_title = "Components",
        yaxis_title = f"{intensity_metric_name} Intensity",
        height=500
    )
    
    return fig

def create_attribution_summary_table(review_data_output, review_dates, intensity_metric, sector_column='Sector'):
    """
    Creates a summary table of attribution results for all review dates
    
    Parameters:
    -----------
    review_data_output : pandas.DataFrame
        Combined review data with portfolio and benchmark data
    
    review_dates : list
        List of review dates to analyze
    
    intensity_metric : str
        The intensity metric to analyze
    
    sector_column : str, default='Sector'
        Column containing sector classifications
    
    Returns:
    --------
    pandas.DataFrame with attribution summary
    """
    # Calculate attribution for each review date
    summaries = []
    for date in review_dates:
        attribution = perform_brinson_attribution(
            review_data_output, 
            date, 
            intensity_metric, 
            sector_column
        )
        
        summaries.append({
            'Review Date': attribution['review_date'],
            'Portfolio Intensity': attribution['portfolio_intensity'],
            'Benchmark Intensity': attribution['benchmark_intensity'],
            'Intensity Difference': attribution['intensity_difference'],
            'Allocation Effect': attribution['allocation_effect'],
            'Selection Effect': attribution['selection_effect'],
            'Interaction Effect': attribution['interaction_effect'],
            'Total Effect': attribution['allocation_effect'] + attribution['selection_effect'] + attribution['interaction_effect']
        })
    
    return pd.DataFrame(summaries)

def display_sector_attribution(attribution_results, intensity_metric_name):
    """
    Display sector attribution effects with vertical grouped bars
    
    Parameters:
    -----------
    attribution_results : dict
        Results from perform_brinson_attribution
        
    intensity_metric_name : str
        Name of the intensity metric for display
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import streamlit as st
    
    # Get sector data
    sector_data = attribution_results['sector_data'].copy()
    
    # Sort sectors by total effect (absolute value)
    sector_data = sector_data.sort_values(by='total_effect', key=abs, ascending=False)
    
    # Extract sectors as a list for the x-axis
    sectors = sector_data['Sector'].tolist()
    
    # Create the plot using make_subplots
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(
        go.Bar(
            x=sectors,
            y=sector_data['allocation_effect'],
            name="Allocation Effect",
            marker_color='#1f77b4'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=sectors,
            y=sector_data['selection_effect'],
            name="Selection Effect",
            marker_color='#ff7f0e'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=sectors,
            y=sector_data['interaction_effect'],
            name="Interaction Effect",
            marker_color='#2ca02c'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=sectors,
            y=sector_data['total_effect'],
            name="Total Effect",
            mode='markers',
            marker=dict(color='black', size=8)
        )
    )
    
    # Format the review date if available
    if 'review_date' in attribution_results:
        review_date = attribution_results['review_date']
        if hasattr(review_date, 'strftime'):
            review_date_text = f" for {review_date.strftime('%Y-%m-%d')}"
        else:
            review_date_text = f" for {review_date}"
    else:
        review_date_text = ""
    
    fig.update_layout(
        barmode='group',
        title=f"Sector-Level {intensity_metric_name} Attribution{review_date_text}",
        xaxis_title="Sector",
        yaxis_title=f"Contribution to {intensity_metric_name} Intensity Difference",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
def brinson_attribution_dashboard(review_data_output, achieved_targets_df, selected_metric, selected_date):
    """
    Simplified function to display the Brinson intensity attribution dashboard
    
    Parameters:
    -----------
    review_data_output : DataFrame
        Combined review data with portfolio and benchmark data
    
    sustainable_factors : list
        List of sustainable factors available for analysis
    """
    
    # Get sector column (adaptable to different dataset structures)
    sector_column = 'Sector' if 'Sector' in review_data_output.columns else next(
        (col for col in review_data_output.columns if 'sector' in col.lower()), None
    )
    
    if not sector_column:
        st.error("No sector column found in the data. Please ensure your data has a sector classification column.")
        return
    
    # Calculate attribution for the selected date (auto-update)
    attribution_results = perform_brinson_attribution(
        review_data_output, 
        selected_date, 
        selected_metric, 
        sector_column
    )
    
    # Create and display the waterfall chart
    # Add explanatory text
    with st.expander("**Curious about Intensity Attribution Analysis ?**"):
        st.markdown("""
        ### Intensity Attribution Analysis Explained
        
        This analysis decomposes the difference between your portfolio's intensity and the benchmark intensity into three components:
        
        **1. Allocation Effect**: This measures the impact of having different sector weights than the benchmark. 
        A negative allocation effect (green) means your sector allocation decisions have reduced intensity compared to the benchmark.
        
        **2. Selection Effect**: This measures the impact of selecting assets with different intensity profiles within each sector.
        A negative selection effect (green) means your security selection within sectors has reduced intensity compared to the benchmark.
        
        **3. Interaction Effect**: This captures the combined impact of your allocation and selection decisions.
        
        ### Interpretation
        
        - **Negative values** (green) represent intensity reduction compared to benchmark (good for environmental metrics)
        - **Positive values** (red) represent intensity increase compared to benchmark (bad for environmental metrics)
        - The **Sector Breakdown** shows which sectors contribute most to your portfolio's intensity difference
        
        This decomposition helps you understand whether your sustainability performance comes primarily from sector allocation decisions or from selecting better-performing companies within sectors.
        """)
    waterfall_fig = plot_attribution_waterfall(attribution_results, selected_metric)
    st.plotly_chart(waterfall_fig, use_container_width=True)
    
    # Tabs for different analyses
    st.markdown("> 🌿 **Tip:** Check out the **Intensities Trajectories** tab to follow your portfolio’s sustainable progress over time!")

    # Create tabs for different analyses

    tab1, tab2, tab3 = st.tabs(["**Cumulative Contributions**", "**Sector Contribution**", "**Intensities Trajectories**"])

    with tab1:
        # Get all attribution results for the selected metric across all dates
        all_dates = sorted(review_data_output["Review Date"].unique())
        attribution_data_list = []
     
        for date in all_dates:
            attr_result = perform_brinson_attribution(
                review_data_output, 
                date, 
                selected_metric, 
                sector_column
            )
            attribution_data_list.append(attr_result)
     
        display_cumulative_attribution(attribution_data_list, selected_metric)

    with tab2:
        
        display_sector_attribution(attribution_results, selected_metric)
    
    with tab3:
        
        footprint.display_targets_evolution(achieved_targets_df, selected_metric)
        
        

def display_cumulative_attribution(attribution_data_list, selected_metric):
    """
    Display cumulative attribution effects over time
    
    Parameters:
    -----------
    attribution_data_list : list of dict
        List of attribution result dictionaries, each containing data for a review date
    
    selected_metric : str
        The intensity metric name for display purposes
    """

    # Convert list of attribution dictionaries to DataFrame
    attribution_df = pd.DataFrame(attribution_data_list)
    
    # Ensure review_date is a datetime for proper sorting
    if not pd.api.types.is_datetime64_any_dtype(attribution_df['review_date']):
        attribution_df['review_date'] = pd.to_datetime(attribution_df['review_date'])
    
    # Sort by date to ensure chronological order
    attribution_df = attribution_df.sort_values('review_date')
    
    # Calculate cumulative effects
    attribution_df['cumulative_allocation'] = attribution_df['allocation_effect'].cumsum()
    attribution_df['cumulative_selection'] = attribution_df['selection_effect'].cumsum()
    attribution_df['cumulative_interaction'] = attribution_df['interaction_effect'].cumsum()
    attribution_df['cumulative_total'] = attribution_df['cumulative_allocation'] + attribution_df['cumulative_selection'] + attribution_df['cumulative_interaction']
    
    
    # 3. Create a bar chart showing progression of cumulative effects
    fig = go.Figure()
    
    review_dates = attribution_df['review_date']
    
    # Add bars for each cumulative effect
    fig.add_trace(go.Bar(
        x=review_dates,
        y=attribution_df['cumulative_allocation'],
        name='Cumulative Allocation',
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        x=review_dates,
        y=attribution_df['cumulative_selection'],
        name='Cumulative Selection',
        marker_color='#ff7f0e'
    ))
    
    fig.add_trace(go.Bar(
        x=review_dates,
        y=attribution_df['cumulative_interaction'],
        name='Cumulative Interaction',
        marker_color='#2ca02c'
    ))
    
    fig.add_trace(go.Scatter(
        x=review_dates,
        y=attribution_df['cumulative_total'],
        mode='lines+markers',
        name='Cumulative Total',
        line=dict(color='black', width=2),
        marker=dict(size=8)
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Progression of Cumulative {selected_metric} Attribution Effects",
        xaxis_title="Review Date",
        yaxis_title="Cumulative Effect",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450
    )
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=attribution_df['review_date'].min(),
        x1=attribution_df['review_date'].max(),
        y0=0,
        y1=0,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    # Display the grouped bar chart
    st.plotly_chart(fig, use_container_width=True)