"""

module for loading the river and rainfall data and doing some basic checks


"""

import json
import pandas as pd
from feature_engine.timeseries.forecasting import LagFeatures
import statsmodels.api as sm
import numpy as np

def load_section(section_name, file="../data/data_extracted.json"):
    with open(file, "r") as f:
        data = json.load(f)

    sections = data["sections"]
    rainfall_stations = data["rainfall_stations"]
    rainfall_data = data["rainfall_data"]
    gauge_stations = data["gauge_stations"]
    gauge_data = data["gauge_data"]

    section = sections[section_name]
    gauge_id = section["gauge_id"]
    rainfall_station_id = section["rainfall_station_id"]

    rainfall = rainfall_data[str(rainfall_station_id)]
    river_level = gauge_data[str(gauge_id)]

    rainfall_df = pd.DataFrame(rainfall).set_index("time")
    river_level_df = pd.DataFrame(river_level).set_index("time")

    df = pd.concat([rainfall_df, river_level_df], axis=1).dropna()
    df.index = pd.to_datetime(df.index)

    return df

class RainImpulseResponse():
    def __init__(self, max_filter_size:int=200):
        self.max_filter_size = max_filter_size
        self.filters = {}
        # self.filter_transforms={"x":lambda x:x,
        #                         "x^2":lambda x:x**2,
        #                         "x^3":lambda x:x**3,
        #                         "exp(x)":lambda x:np.exp(x)}
        self.filter_transforms={"x":lambda x:x,
                                "x^3": lambda x: x ** 3,
                                "10^x":lambda x:10**x}

    def fit(self, data:list):
        d_all = pd.concat(data, axis=0)
        max_level = d_all["level"].max()
        min_level = d_all["level"].min()
        max_rain = d_all["rain"].max()
        min_rain = d_all["rain"].min()

        for k, v in self.filter_transforms.items():
            filt_t = []
            for d in data:
                if len(d) >= 5*self.max_filter_size:
                    rain = (d["rain"].to_numpy() - min_rain)/(max_rain - min_rain)
                    level = (d["level"].to_numpy() - min_level)/(max_level - min_level)
                    ccf = sm.tsa.stattools.ccf(v(level), rain)
                    filt_t.append(self._clean_filter(ccf, max_filter_size=self.max_filter_size))

            self.filters[k] = np.mean(np.vstack(filt_t), axis=0)

    def transform(self, data):
        transformed_data = []
        for d in data:
            tmp = d.copy()
            rain = d["rain"].to_numpy()
            for name, filter in self.filters.items():
                tmp['rain_impulse'+'_'+name] = np.convolve(tmp["rain"].to_numpy(), filter, mode="full")[:len(rain)]
            transformed_data.append(tmp)

        return transformed_data

    @staticmethod
    def _clean_filter(filter:np.ndarray, max_filter_size:int=200) -> np.ndarray:
        filter = filter[:max_filter_size].clip(0)
        i_max = np.argmax(filter)
        for i in range(i_max + 1, len(filter)):
            filter[i] = min(filter[i], filter[i - 1])

        for i in range(i_max, 0, -1):
            filter[i - 1] = min(filter[i], filter[i - 1])

        return filter




def split_contiguous(df):
    df['frame'] = (df.index.to_series().diff().dt.seconds > 60 * 60).cumsum()
    list_of_dfs = []
    for ct, data in df.groupby('frame'):
        list_of_dfs.append(data)
    return list_of_dfs



