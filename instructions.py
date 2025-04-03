import streamlit as st

def display_instructions_page():
    # Description using markdown for better formatting
    st.markdown("""
    **Welcome to the Sustainable Portfolio Design Application!**
    
    This application built in Python enables you to design investment portfolios that align with your sustainability objectives—no coding required! The S&P 500 is used as the benchmark, with data from 2018 to present, sourced from YFinance. Sustainability scores for each constituent are synthetically generated based on their sector classification.

    This tool employs the **Target Exposure methodology**, where sustainability data is treated as factors. The goal is to tilt portfolio weights to satisfy these constraints. To achieve this, we determine the exponent that indicates how much to tilt the weights to meet your sustainability exposure targets.

    ### How to Use the Application:
    
    1. **Select Constraints**:  
       Head over to the **Select Constraints** page in the sidebar to define the sustainability factors you wish to include in your portfolio. You can specify the minimum reduction for each factor and the trajectory for more stringent criteria. Additionally, you can set weight limits and exclude specific sub-sectors from your index.
    
    2. **Generate Weights**:  
       Once you have set your constraints, navigate to the **Generate Weights** page. Here, you will find two tabs:
       - **Financial Analysis**: Dive into key financial metrics for the portfolio.
       - **Sustainability Analysis**: Explore the sustainability characteristics of your portfolio and compare them to the benchmark.

    After generating the portfolio weights, assess how well the portfolio aligns with your sustainability goals. If you are satisfied with the results, **you can easily export the portfolio weights to Excel**.
    
    **Ready to design your sustainable portfolio? Start now with just a few clicks!**
    """)

