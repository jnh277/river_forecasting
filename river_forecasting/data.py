"""

module for loading the river and rainfall data and doing some basic checks

"""

import json
import pandas as pd


def load_section(section_name, file="../data/data_extracted.json"):
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

def load_training_data(section_name, file="../data/data_extracted.json") -> list[pd.DataFrame]:
    df = load_section(section_name, file=file)
    dfs = split_contiguous(df)
    for df in dfs:
        df["level_diff"] = df["level"].diff()
    return dfs


def split_contiguous(df):
    df['frame'] = (df.index.to_series().diff().dt.seconds > 60 * 60).cumsum()
    list_of_dfs = []
    for ct, data in df.groupby('frame'):
        list_of_dfs.append(data.drop(columns=["frame"]))
    return list_of_dfs

