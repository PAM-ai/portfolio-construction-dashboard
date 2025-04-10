"""
Selection Page for Sustainability Constraints in the Sustainable Index Dashboard.
"""

import streamlit as st
import os
import pandas as pd

def go_to(page_name):
    """Helper function to navigate between pages."""
    st.session_state.page = page_name
    st.rerun()

@st.cache_data
def load_data():
    """
    Load the required data for index construction.

    Returns:
        tuple: DataFrames containing review data and price data with weights.
    """
    path = os.path.join(os.getcwd(), "data")
    review_data = pd.read_csv(os.path.join(path, "review_data.csv"))
    prices = pd.read_csv(os.path.join(path, "prices_with_weights.csv"))
    return review_data, prices

def selection_page():
    """
    Selection page where users define sustainability constraints 
    and configure portfolio settings.

    The function allows users to:
    - Select sustainability factors for constraints.
    - Define target reductions and annual reduction rates.
    - Configure portfolio construction parameters.
    - Exclude specific sub-sectors from the portfolio.
    """

    st.title("Sustainable Index Construction Dashboard - Configure Portfolio")

    with st.sidebar.expander("🧩 **Tutorial: How to Configure Your Sustainable Portfolio**", expanded=True):
        st.markdown("""
        1. **Select Sustainability Factors**  
        2. **Set Reduction Targets**  
        3. **Configure Portfolio Parameters** *(Sidebar)*  
        4. **Exclude Sub-Sectors** *(Optional)*  
        """)


    with st.spinner("Loading data..."):
        review_data, prices = load_data()
    
    # Store loaded data in session state
    st.session_state["review_data"] = review_data
    st.session_state["prices"] = prices

    # Define available sustainability factors
    sustainable_factors = ["Carbon Emissions", "Waste", "Water Usage"]

    # Initialize session state for constraints and config if not already set
    st.session_state.setdefault('constraints', {})
    st.session_state.setdefault('config', {})

    # --- STEP 1: SELECT CONSTRAINTS ---

    st.markdown("#### Targets Selection")

    if st.button("**Press here to Generate weights** ➡️", type="secondary", use_container_width=True):
      go_to("Generate Weights")

    # User selects which sustainability factors to constrain
    selected_items = st.multiselect("Select sustainability factors to constrain :", sustainable_factors, default=["Carbon Emissions", "Water Usage"])

    # Reset constraints if selection changes
    st.session_state['constraints'] = {}

    # Iterate over selected sustainability factors to configure constraints
    for item in selected_items:
        with st.expander(f"Configure {item} Constraint", expanded=True):  
            col1, col2 = st.columns([2, 2])  # Two-column layout for better alignment

            with col1:
                # User inputs target reduction percentage
                target_value = st.number_input(
                    f"Target Reduction (%) for {item}",
                    min_value=0, max_value=50, value=30, step=5
                )

                # Checkbox for enabling annual reduction rate
                enable_annual_reduction = st.checkbox(f"Enable Annual Reduction for {item}", value=False)

            # Default annual reduction rate to 0 unless user enables it
            annual_reduction_rate = 0
            if enable_annual_reduction:
                with col2:
                    annual_reduction_rate = st.number_input(
                        f"Annual Reduction (%) for {item}",
                        min_value=0.0, max_value=10.0, value=5.0, step=0.5
                    )

            # Store constraints in session state with proper sign convention
            st.session_state['constraints'][item] = {
                'target_reduction': - (target_value / 100),  # Negative for reduction
                'annual_reduction_rate': - (annual_reduction_rate / 100)  
            }

    # --- STEP 2: PORTFOLIO CONFIGURATION ---
    st.sidebar.subheader("Portfolio Construction Parameters")
    
    # Portfolio configuration parameters
    st.session_state['config']['Capacity Ratio'] = st.sidebar.number_input(
        "Capacity Ratio", min_value=10, value=10, step=1
    )
    st.session_state['config']['Max Weight'] = st.sidebar.number_input(
        "Max Weight", min_value=0.05, value=0.05, step=0.01,
    )
    st.session_state['config']['Stock Bound'] = st.sidebar.number_input(
        "Stock Bound", min_value=1, value=1, step=1
    )

    # --- STEP 3: SUB-SECTOR EXCLUSIONS ---
    try:
        sub_sectors_data = review_data[['Sector', 'Sub-Sector']].drop_duplicates().dropna()
    except Exception as e:
        st.error(f"Error loading sub-sectors: {e}")
        sub_sectors_data = pd.DataFrame(columns=["Sector", "Sub-Sector"])

    st.markdown("#### Sub-Sector Exclusions (Type below to search for and exclude sub-sectors)")

    excluded_sub_sectors = st.multiselect("Select SubSectors to exclude from your index:", list(sub_sectors_data["Sub-Sector"].unique()), default="Oil & Gas Integrated")

    # Store excluded sub-sectors in session state
    st.session_state['excluded_sub_sectors'] = excluded_sub_sectors