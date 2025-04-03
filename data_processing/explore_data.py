for i, date in enumerate(['2018', "2019"]):
    print(i, date)

r = 0.93**(1/2) - 1

(1 + r)**2 - 1

import streamlit as st
import os
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

@st.cache_data
def load_data():
    """Function to load required data"""
    path = os.path.join(os.getcwd(), "data")
    return (
        pd.read_csv(os.path.join(path, "review_data.csv")),
        pd.read_csv(os.path.join(path, "prices_with_weights.csv"))
    )

def selection_page():
    """Selection page for sustainable constraints and configuration parameters"""

    with st.spinner("Loading data"):
        review_data, prices = load_data()

    sustainable_factors = ["Carbon Emissions", "Waste", "Water Usage"]
    
    st.header("Step 1: Select Constraints and Configure Portfolio")
    
    # Initialize session state for constraints and config
    if 'constraints' not in st.session_state:
        st.session_state['constraints'] = {}
    if 'config' not in st.session_state:
        st.session_state['config'] = {}
    if 'excluded_sub_sectors' not in st.session_state:
        st.session_state['excluded_sub_sectors'] = []

    # Select sustainability constraints
    selected_items = st.multiselect("Select sustainability factors to constrain", sustainable_factors)

    # Reset constraints when the selected items change
    st.session_state['constraints'] = {}  # Clear previous constraints

    # Set reduction targets and annual reduction rates for selected constraints
    for item in selected_items:
        # Use columns for side-by-side layout
        col1, col2 = st.columns([2, 1])  # Adjust the width ratio if necessary

        # Target Reduction input
        with col1:
            target_value = st.number_input(
                f"Enter target reduction (%) for {item}", min_value=0, max_value=100, value=30, step=5
            )
        
        # Checkbox for enabling annual reduction rate input
        with col2:
            enable_annual_reduction = st.checkbox(
                f"Enable annual reduction rate for {item}", value=False
            )
        
        # Annual reduction input appears if checkbox is selected
        annual_reduction_rate = 0
        if enable_annual_reduction:
            with col2:
                annual_reduction_rate = st.number_input(
                    f"Enter annual reduction rate (%) for {item} (e.g., 5 for 5%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1
                )

        # Store both target reduction and annual reduction rate
        st.session_state['constraints'][item] = {
            'target_reduction': - (target_value / 100),  # Negative for reduction
            'annual_reduction_rate': - (annual_reduction_rate / 100)  # Store annual reduction as decimal
        }

    # Portfolio Configuration Parameters
    st.subheader("Portfolio Construction Parameters")
    st.session_state['config']['Capacity Ratio'] = st.number_input("Capacity Ratio", min_value=1, value=10, step=1)
    st.session_state['config']['Max Weight'] = st.number_input("Max Weight", min_value=0.01, value=0.05, step=0.01)
    st.session_state['config']['Stock Bound'] = st.number_input("Stock Bound", min_value=0, value=1, step=1)
    
    # Load sub-sectors and sectors from weights_data_clean.csv
    try:
        sub_sectors_data = review_data[['Sector', 'Sub-Sector']].drop_duplicates().dropna()
    except Exception as e:
        st.error(f"Error loading sub-sectors: {e}")
        sub_sectors_data = pd.DataFrame(columns=["Sector", "Sub-Sector"])
    
    # Exclude sub-sectors using AgGrid
    st.subheader("Sub-Sector Exclusions")
    sub_sectors_data["Exclude"] = False
    
    gb = GridOptionsBuilder.from_dataframe(sub_sectors_data)
    gb.configure_column("Exclude", editable=True, cellEditor='agCheckboxCellEditor')
    gb.configure_grid_options(domLayout='autoHeight')  # Ensure it adjusts properly
    gb.configure_side_bar()  # Enable sidebar for filtering/searching
    grid_options = gb.build()
    
    grid_response = AgGrid(
        sub_sectors_data,
        gridOptions=grid_options,
        update_mode='MANUAL',
        fit_columns_on_grid_load=True  # Ensures table columns use the full space
    )
    
    st.session_state['excluded_sub_sectors'] = grid_response['data'][grid_response['data']['Exclude'] == True]['Sub-Sector'].tolist()
    
    # Display selected settings
    st.write("### Selected Constraints")
    st.json(st.session_state['constraints'])
    
    st.write("### Portfolio Configuration")
    st.json(st.session_state['config'])
    
    st.write("### Excluded Sub-Sectors")
    st.write(st.session_state['excluded_sub_sectors'])