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
    """Selection page for sustainability constraints"""

    with st.spinner("Loading data"):
        review_data, prices = load_data()
    
    st.session_state["review_data"] = review_data
    st.session_state["prices"] = prices
    
    sustainable_factors = ["Carbon Emissions", "Waste", "Water Usage"]
    
    st.header("Step 1: Select Constraints and configure Portfolio")
    
    # Initialize session state for constraints and config if not already set
    if 'constraints' not in st.session_state:
        st.session_state['constraints'] = {}

    if 'config' not in st.session_state:
        st.session_state['config'] = {}  # Ensure 'config' exists before modifying it

    # Select sustainability constraints
    selected_items = st.multiselect("Select sustainability factors to constrain", sustainable_factors)

    # Reset constraints if selection changes
    st.session_state['constraints'] = {}

    # Iterate over selected items to define constraints
    for item in selected_items:
        with st.expander(f"Configure {item} Constraint", expanded=True):  # Expander for a cleaner UI
            col1, col2 = st.columns([2, 2])  # Define two columns for alignment

            # Target Reduction Input
            with col1:
                target_value = st.number_input(
                    f"Target Reduction (%) for {item}",
                    min_value=0, max_value=100, value=30, step=5
                )

                # Checkbox + Annual Reduction Rate (only shown if checked)
                enable_annual_reduction = st.checkbox(
                    f"Enable Annual Reduction for {item}", value=False
                )

            annual_reduction_rate = 0  # Default to zero for 

            if enable_annual_reduction:
                with col2:
                    annual_reduction_rate = st.number_input(
                        f"Annual Reduction (%) for {item}",
                        min_value=0.0, max_value=100.0, value=5.0, step=0.1
                    )

            # Store constraints in session state with correct sign adjustments
            st.session_state['constraints'][item] = {
                'target_reduction': - (target_value / 100),  # Ensure negative sign
                'annual_reduction_rate': - (annual_reduction_rate / 100)  # Ensure negative sign
            }

    # Portfolio Configuration Parameters
    st.sidebar.subheader("Portfolio Construction Parameters")
    st.session_state['config']['Capacity Ratio'] = st.sidebar.number_input("Capacity Ratio", min_value=1, value=10, step=1)
    st.session_state['config']['Max Weight'] = st.sidebar.number_input("Max Weight", min_value=0.01, value=0.05, step=0.01)
    st.session_state['config']['Stock Bound'] = st.sidebar.number_input("Stock Bound", min_value=0, value=1, step=1)
    
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

    st.write("### Portfolio Configuration")
    st.json(st.session_state['config'])
    
    st.write("### Excluded Sub-Sectors")
    st.write(st.session_state['excluded_sub_sectors'])

    # Display updated constraints
    st.subheader("Final Constraints:")
    st.json(st.session_state['constraints'])
