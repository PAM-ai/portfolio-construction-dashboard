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
    and configure portfolio settings, with improved guidance and clarity.
    """

    st.title("Sustainable Index Construction Dashboard")

    with st.spinner("Loading data..."):
        review_data, prices = load_data()

    st.session_state["review_data"] = review_data
    st.session_state["prices"] = prices

    sustainable_factors = ["Carbon Emissions", "Waste", "Water Usage"]

    # Initialize session state
    st.session_state.setdefault('constraints', {})
    st.session_state.setdefault('config', {})

    # Set Stock Bound automatically to 1 in session state (no user trace)
    st.session_state['config']['Stock Bound'] = 1

    # --- Sidebar: Portfolio Parameters and Exclusions ---
    with st.sidebar:
        st.header("⚙️ Portfolio Parameters")

        st.session_state['config']['Capacity Ratio'] = st.number_input(
            "Capacity Ratio",
            min_value=10, value=10, step=1,
            help="""Controls the overall scaling capacity of the portfolio.
- It multiplies each stock's starting weight.
- Higher values give the solver more flexibility to increase weights of all stocks, within other limits.
Example: if initial weight is 0.2%, and capacity is 10, max from this rule is 2%."""
        )

        st.session_state['config']['Max Weight'] = st.number_input(
            "Max Weight",
            min_value=0.05, value=0.05, step=0.01,
            help="""Absolute limit for any single stock.
No stock in the portfolio will exceed this weight, regardless of other settings.
Acts as a hard cap to prevent concentration risk.
Example: 0.05 means no stock exceeds 5%."""
        )

        st.markdown("---")
        st.header("🚫 Sub-Sector Exclusions")
        st.markdown("Optionally, remove entire sub-sectors from your index. **Type in the name to filter**.")

        try:
            sub_sectors_data = review_data[['Sector', 'Sub-Sector']].drop_duplicates().dropna()
        except Exception as e:
            st.error(f"Error loading sub-sectors: {e}")
            sub_sectors_data = pd.DataFrame(columns=["Sector", "Sub-Sector"])

        excluded_sub_sectors = st.multiselect(
            "Select sub-sectors to exclude:",
            list(sub_sectors_data["Sub-Sector"].unique()),
            default=["Oil & Gas Integrated"],
            help="Sub-Sectors you want to completely remove from your index construction."
        )

        st.session_state['excluded_sub_sectors'] = excluded_sub_sectors

    # ---- Selection first, for logic dependencies ----
    st.header("🎯 Set Sustainability Targets")
    
    # Initialize selected_items with defaults if not already in session state
    if 'selected_items' not in st.session_state:
        st.session_state['selected_items'] = ["Carbon Emissions", "Water Usage"]
    
    st.markdown("Choose the environmental factors you want to target in your portfolio:")

    # Use a different approach - store selections directly in session state
    # This avoids the need for a callback that might cause double-click issues
    selected_items = st.multiselect(
        "Select sustainability factors to constrain:",
        sustainable_factors,
        key="selected_items",  # Direct key to session state
        help="You can select multiple factors to solve for simultaneously."
    )
    
    # Check if selection is valid based on the direct session state key
    is_selection_valid = bool(st.session_state["selected_items"])
    
    # Button for portfolio weight generation
    if st.button("➡️ Generate Portfolio Weights", type="primary", use_container_width=True, disabled=not is_selection_valid):
        if is_selection_valid:
            go_to("Generate Weights")
    
    # User guidance based on selection state
    if not is_selection_valid:
        st.info("👇 Please select at least one sustainability factor above to enable portfolio generation.")
    else:
        st.success("✅ You've selected sustainability factors. Configure your targets below, then click the button above when ready.")
    
    # Update selected_items from the multiselect for the rest of the function
    selected_items = st.session_state["selected_items"]
    
    # Track previous selection for constraint management
    previous_selection = st.session_state.get('previous_selected_items', [])
    if selected_items != previous_selection:
        st.session_state['constraints'] = {}
    st.session_state['previous_selected_items'] = selected_items

    # ---- Now, show the expanders etc. with minimal gap ----
    if selected_items:
        st.markdown("#### Define reduction goals for each selected factor:")

    # Default targets
    default_target_values = {
        "Carbon Emissions": 30,
        "Water Usage": 20,
        "Waste": 20
    }

    defaut_trajectory_values = {
        "Carbon Emissions": 5.0,
        "Water Usage": 0.0,
        "Waste": 0.0
    }

    default_trajectory_item = ["Carbon Emissions"]

    for item in selected_items:
        with st.expander(f"🌿 Configure {item} Constraints", expanded=True):
            col1, col2 = st.columns([1, 1])

            with col1:
                target_value = st.number_input(
                    "🎯 Target Reduction (%)",
                    min_value=0, max_value=50,
                    value=default_target_values.get(item),
                    step=5,
                    help="Overall reduction you aim to achieve compared to the current baseline.",
                    key=f"target_{item}"
                )

            with col2:
                annual_reduction_rate = st.number_input(
                        "📉 Annual Reduction (%)",
                        min_value=0.0, max_value=10.0,
                        value=defaut_trajectory_values.get(item), step=0.5,
                        help="Specify yearly reduction to achieve a gradual transition. The baseline is the 2018 Benchmark Intensity.",
                        key=f"rate_{item}"
                    )

            # Save to session state
            st.session_state['constraints'][item] = {
                'target_reduction': - (target_value / 100),
                'annual_reduction_rate': - (annual_reduction_rate / 100)
            }


    # Small container for explain functionality
    
    if selected_items:
        with st.sidebar.expander("🧩 View summary of my current choices"):
            if st.session_state['constraints']:
                explanations = []
                for factor, settings in st.session_state['constraints'].items():
                    explanations.append(
                        f"- **{factor}**: Target {abs(settings['target_reduction']*100)}% reduction"
                        + (f", Annual {abs(settings['annual_reduction_rate']*100)}%" if settings['annual_reduction_rate'] != 0 else "")
                    )

                config = st.session_state['config']
                explanations.append("\nPortfolio constraints:")
                explanations.append(f"- **Capacity Ratio**: Up to {config['Capacity Ratio']}× the starting weights.")
                explanations.append(f"- **Max Weight**: Max {config['Max Weight']*100:.1f}% per stock.")

                st.info("\n".join(explanations))
            else:
                st.warning("You have not configured any sustainability factors yet.")