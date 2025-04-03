"""
Dashboard for creating sustainable indexes using the Target Exposure methodology.
"""

import streamlit as st
from instructions import display_instructions_page
from items_selection import selection_page
from index_generation import index_generation_page

# Set page configuration at the start (only once)
st.set_page_config(page_title="Sustainable Index Construction", layout="wide")

def main():
    """
    Main function to run the Streamlit dashboard.

    This dashboard allows users to construct a sustainable index by:
    1. Selecting constraints based on sustainability factors.
    2. Running the optimization process to generate the final index weights.
    """
    st.title("Sustainable Index Construction Dashboard")

    # Sidebar for navigation
    page = st.sidebar.radio("Navigation", ["Instructions", "Select Constraints", "Generate Weights"])

    if page == "Instructions":
        display_instructions_page() # Calls the instruction page
    if page == "Select Constraints":
        selection_page()  # Calls the constraint selection page
    elif page == "Generate Weights":
        index_generation_page()  # Calls the optimization page

if __name__ == "__main__":
    main()
