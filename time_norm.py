import numpy as np
import copy
from scipy.interpolate import interp1d

def interpol_and_norm(mvc_normed):
    mvc_normed_ = {}

    for key, sensors in mvc_normed.items():
        mvc_normed_[key] = [list(sensor) for sensor in sensors]  # Shallow copy of nested arrays
        for sensor, data in enumerate(mvc_normed_[key]):
            for num, sensor_data in enumerate(data):
                x = np.linspace(0, len(sensor_data)-1, len(sensor_data))
                cubic_fx = interp1d(x, sensor_data, kind='cubic')
                resampled_x = np.linspace(0, len(sensor_data)-1, 101)
                mvc_normed_[key][sensor][num] = cubic_fx(resampled_x)

    return mvc_normed_
