import pandas as pd
import os
import matplotlib.pyplot as plt

from river_forecasting.model_manager import RegressionModelType

SECTION_NAME = "shoalhaven-river-oallen-ford-to-tallowa-dam"

filepath = os.path.join("../models/",SECTION_NAME,"model_info.csv")
df = pd.read_csv(filepath, index_col="Unnamed: 0")

# plt.plot(df["forecast_step"], df[""])
gb = df.groupby(by="regression_model_type")
for type, g in gb:
    plt.plot(g["forecast_step"], g["test score"], label=type)
    # g.plot(x='forecast_step', y='test score', label=type)

plt.legend()
plt.title(SECTION_NAME)
plt.xlabel("Forecast time ahead (hours)")
plt.ylabel("test R score (1 is best, 0 worst)")
plt.show()

for type, g in gb:
    plt.plot(g["forecast_step"], g["test mse"], label=type)
    # g.plot(x='forecast_step', y='test score', label=type)

plt.legend()
plt.title(SECTION_NAME)
plt.xlabel("Forecast time ahead (hours)")
plt.ylabel("test mse (low is best)")
plt.show()

