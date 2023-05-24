import numpy as np    
    
def max_of_sensor(data_dict):
    max_arr = []
    
    for trial_data in data_dict.values():
        trial_max = [np.max(np.concatenate(sensor_data)) for sensor_data in trial_data]
        max_arr.append(trial_max)
    
    max_arr = np.array(max_arr)
    sensor_max = np.max(max_arr, axis=0)

    return sensor_max