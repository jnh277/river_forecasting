"""

module for loading the river and rainfall data and doing some basic checks

"""

import json
import pandas as pd
import os

DATA_DIRECTORY = "../data"



def _load_section_water_data_online(section_name: str, file: str) -> pd.DataFrame:

    filename = os.path.join(DATA_DIRECTORY, "water_data_online", "level_" + section_name + ".csv")
    levels = pd.read_csv(filename, header=9, parse_dates=["#Timestamp"]).dropna()
    filename = os.path.join(DATA_DIRECTORY, "water_data_online", "rainfall_" + section_name + ".csv")
    rainfall = pd.read_csv(filename, header=9, parse_dates=["#Timestamp"]).dropna()

    rainfall.rename(columns={"#Timestamp": "datetime",
                             "Value": "rain"}, inplace=True)

    rainfall = rainfall[["datetime", "rain"]].set_index("datetime")
    # resample to hourly
    rainfall = rainfall.groupby(by=rainfall.index.ceil("1h")).sum()

    levels.rename(columns={"#Timestamp": "datetime",
                           "Value": "level"}, inplace=True)
    levels = levels[["datetime", "level"]].set_index("datetime")
    levels = levels.groupby(by=levels.index.floor("1h")).first(1)
    data = pd.concat([levels, rainfall], axis=1).dropna()

    return data



def load_section(section_name, source: str="wikiriver", **kwargs) -> pd.DataFrame:
    if source=="wikiriver":
        df = _load_section_wikiriver(section_name, **kwargs)
    elif source=="waterdataonline":
        df = _load_section_water_data_online(section_name, **kwargs)
    return df

def _load_section_wikiriver(section_name: str,
                            file: str = "../data/data_extracted.json") -> pd.DataFrame:
    with open(file, "r") as f:
        data = json.load(f)

    sections = data["sections"]
    rainfall_data = data["rainfall_data"]
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

def load_training_data(section_name, file="../data/data_extracted.json", source="wikiriver") -> list[pd.DataFrame]:
    df = load_section(section_name, file=file, source=source)
    dfs = split_contiguous(df)
    for df in dfs:
        df["level_diff"] = df["level"].diff()
    return dfs


def split_contiguous(df: pd.DataFrame) -> list[pd.DataFrame]:
    df['frame'] = (df.index.to_series().diff().dt.seconds > 60 * 60).cumsum()
    list_of_dfs = []
    for ct, data in df.groupby('frame'):
        if len(data) > 300:
            list_of_dfs.append(data.drop(columns=["frame"]))
    return list_of_dfs

if __name__=="__main__":
    import matplotlib.pyplot as plt

    filename = "../data/water_data_online/level_franklin_at_fincham.csv"
    levels = pd.read_csv(filename, header=9, parse_dates=["#Timestamp"]).dropna()
    filename = "../data/water_data_online/rainfall_franklin_at_fincham.csv"
    rainfall = pd.read_csv(filename, header=9, parse_dates=["#Timestamp"]).dropna()

