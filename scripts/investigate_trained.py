import pandas as pd
import os
import matplotlib.pyplot as plt

from river_forecasting.model_manager import RegressionModelType

# SECTION_NAME = "shoalhaven-river-oallen-ford-to-tallowa-dam"
SECTION_NAME = "collingwood_below_alma"
# SECTION_NAME = "franklin_at_fincham"
# SECTION_NAME = "franklin_at_fincham_long"

filepath = os.path.join("../models/",SECTION_NAME,"model_info.csv")
df_all = pd.read_csv(filepath, index_col="Unnamed: 0")


df_reg = df_all[df_all["loss type"]=="SQUARED_ERROR"]
# plt.plot(df["forecast_step"], df[""])
gb = df_reg.groupby(by="regression_model_type")
for type, g in gb:
    plt.plot(g["forecast_step"], g["test score"], label=type)
    # g.plot(x='forecast_step', y='test score', label=type)

plt.legend()
plt.title(SECTION_NAME)
plt.xlabel("Forecast time ahead (hours)")
plt.ylabel("test R score (1 is best, 0 worst)")
plt.ylim([0, 1])
plt.show()

for type, g in gb:
    plt.plot(g["forecast_step"], g["test mse"], label=type)
    # g.plot(x='forecast_step', y='test score', label=type)

plt.legend()
plt.title(SECTION_NAME)
plt.xlabel("Forecast time ahead (hours)")
plt.ylabel("test mse (low is best)")
plt.show()


for type, g in gb:
    plt.plot(g["forecast_step"], g["test mae"], label=type)
    # g.plot(x='forecast_step', y='test score', label=type)

plt.legend()
plt.title(SECTION_NAME)
plt.xlabel("Forecast time ahead (hours)")
plt.ylabel("test mae (low is best)")
plt.show()


df_quant = df_all[df_all["loss type"]=="QUANTILE"]

gb = df_quant.groupby(by=["regression_model_type", "quantile"])

for type, g in gb:
    plt.plot(g["forecast_step"], g["test pinball"], label=type)
    # g.plot(x='forecast_step', y='test score', label=type)

plt.legend()
plt.title(SECTION_NAME)
plt.xlabel("Forecast time ahead (hours)")
plt.ylabel("test loss")
plt.show()