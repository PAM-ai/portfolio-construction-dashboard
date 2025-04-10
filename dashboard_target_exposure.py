"""
Dashboard for creating sustainable indexes using the Target Exposure methodology.
"""
import streamlit as st
import app_pages.instructions as instructions
import app_pages.items_selection as selection
import app_pages.index_generation as index_generation
from importlib import reload

reload(instructions)
reload(selection)
reload(index_generation)

# Set page configuration at the start (only once)
st.set_page_config(page_title="Sustainable Index Construction", layout="wide")

def main():
    
    # Initialize session state for page if it doesn't exist
    if "page" not in st.session_state:
        st.session_state.page = "Instructions"

    # Sidebar for navigation, linked with session state
    pages = ["Instructions", "Select Constraints", "Generate Weights"]
    selected_page = st.sidebar.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.page)
    )

    # Sync session state with sidebar selection
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

    # Page display logic
    if st.session_state.page == "Instructions":
        instructions.display_instructions_page()

    elif st.session_state.page == "Select Constraints":
        selection.selection_page()

    elif st.session_state.page == "Generate Weights":
        index_generation.index_generation_page()

if __name__ == "__main__":
    main()
