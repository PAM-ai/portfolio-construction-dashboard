import os
import yfinance as yf
import pandas as pd
import numpy as np
import time

path = r"C:\Users\Michel\Documents\Python Projects\Portfolio Construction Basics\Target Exposure\Target Exposure App\data"
cons = pd.read_csv(os.path.join(path, "weights_data_clean.csv"))
symbols = list(cons["Symbol"].unique())

def get_sector_subsector(symbol):
    stock = yf.Ticker(symbol)
    try:
        info = stock.info
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        return sector, industry
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return 'N/A', 'N/A'

# Apply function with delay
sectors_subsectors = []
for symbol in symbols:
    sectors_subsectors.append(get_sector_subsector(symbol))
    time.sleep(2)  # Wait 2 seconds between requests to avoid rate limiting

sectors = pd.DataFrame()
sectors[['Sector', 'Sub-Sector']] = pd.DataFrame(sectors_subsectors, index=symbols)
sectors = sectors.reset_index().rename(columns={"index": "Symbol"})

cons_with_sectors = cons.drop("Sector", axis=1).merge(sectors, on="Symbol", how="left")
cons_with_sectors.info()

# Export to csv
cons_with_sectors.to_csv(os.path.join(path, "weights_data_clean.csv"), index=False)

