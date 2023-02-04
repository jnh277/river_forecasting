"""
        Functions for preparing data for training and prediction



"""
import pandas as pd
import statsmodels.api as sm
import numpy as np



class RainImpulseResponseFilter():
    def __init__(self, max_filter_size: int = 200):
        self.max_filter_size = max_filter_size
        self.filters = {"x":None, "x^3":None, "10^x":None}
        self.filter_names = ["x", "x^3", "10^x"]

    def fit_filter(self, data: list[pd.DataFrame]):
        d_all = pd.concat(data, axis=0)
        max_level = d_all["level"].max()
        min_level = d_all["level"].min()
        max_rain = d_all["rain"].max()
        min_rain = d_all["rain"].min()

        for i in range(3):
            filt_t = []
            for d in data:
                if len(d) >= 5 * self.max_filter_size:
                    rain = (d["rain"].to_numpy() - min_rain) / (max_rain - min_rain)
                    level = (d["level"].to_numpy() - min_level) / (max_level - min_level)
                    level_trans = self.output_transform(level, i)
                    ccf = sm.tsa.stattools.ccf(level_trans, rain)
                    filt_t.append(self._clean_filter(ccf, max_filter_size=self.max_filter_size))

            self.filters[self.filter_names[i]] = np.mean(np.vstack(filt_t), axis=0)

    def output_transform(self, x: np.ndarray, i: int) -> np.ndarray:
        assert 0 <= i < 3, "i must be in range [0, 2]"
        if i == 0:
            return x
        elif i == 1:
            return x ** 3
        elif i == 2:
            return 10 ** x

    def apply_filter(self, data: list[pd.DataFrame]) -> list[pd.DataFrame]:
        transformed_data = []
        for d in data:
            tmp = d.copy()
            rain = d["rain"].to_numpy()
            for name, filter in self.filters.items():
                tmp['rain_impulse' + '_' + name] = np.convolve(tmp["rain"].to_numpy(), filter, mode="full")[:len(rain)]
            transformed_data.append(tmp)

        return transformed_data

    @staticmethod
    def _clean_filter(filter: np.ndarray, max_filter_size: int = 200) -> np.ndarray:
        filter = filter[:max_filter_size].clip(0)
        i_max = np.argmax(filter)
        for i in range(i_max + 1, len(filter)):
            filter[i] = min(filter[i], filter[i - 1])

        for i in range(i_max, 0, -1):
            filter[i - 1] = min(filter[i], filter[i - 1])

        return filter

