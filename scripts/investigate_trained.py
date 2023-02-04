import pandas as pd
import os
import matplotlib.pyplot as plt

SECTION_NAME = "shoalhaven-river-oallen-ford-to-tallowa-dam"

filepath = os.path.join("../models/",SECTION_NAME,"model_info.csv")
df = pd.read_csv(filepath, index_col="Unnamed: 0")

# plt.plot(df["forecast_step"], df[""])
df.plot(x='forecast_step', y='test score')
plt.show()