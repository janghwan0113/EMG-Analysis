import copy
import numpy as np

def mvc_norm(data_dict, max_array):
    data_dict_ = {}

    for key, sensors in data_dict.items():
        data_dict_[key] = [sensor / max_val for sensor, max_val in zip(sensors, max_array)]

    return data_dict_
