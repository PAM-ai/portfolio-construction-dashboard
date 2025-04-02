"""Dashboard for creating sustainable indexes using the Target Exposure methodology"""

# Import libraries
import streamlit as st
from items_selection import selection_page
from index_generation import index_generation_page

# Set page configuration at the start (only once)
st.set_page_config(page_title="Sustainable Index Construction", layout="wide")

def main():
    """Main function to run the Streamlit dashboard"""
    st.title("Sustainable Index Construction")
    st.markdown(
        """
        Use this interactive dashboard to construct a portfolio that matches your sustainability requirements.
        """
    )

    # Sidebar for navigation
    page = st.sidebar.radio("Navigation", ["Select Constraints", "Run Optimization"])

    if page == "Select Constraints":
        selection_page()
    elif page == "Run Optimization":
        index_generation_page()

if __name__ == "__main__":
    main()
