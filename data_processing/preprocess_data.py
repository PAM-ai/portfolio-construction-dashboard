import os
import pandas as pd

path = r"C:\Users\Michel\Documents\Python Projects\Portfolio Construction Basics\Target Exposure\Target Exposure App\data"
cons = pd.read_csv(os.path.join(path, "weights_data_clean.csv"))
sustainable_data = pd.read_csv(os.path.join(path, "sustainable_data.csv"))

review_data = cons.merge(sustainable_data.drop("Sector", axis=1), on=["Review Date", "Symbol"], how="left")
review_data.info()

# Export to csv
review_data.to_csv(os.path.join(path, "review_data.csv"), index=False)