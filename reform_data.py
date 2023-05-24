import numpy as np

def data_reform(data, file_list, sensor_R, sensor_L):

    whole_data = np.empty((0,60))
    
    for same_timepoint_tuple in zip(*data[file_list[0]][sensor_R],
                                    *data[file_list[0]][sensor_L],
                                    *data[file_list[1]][sensor_R],
                                    *data[file_list[1]][sensor_L], 
                                    *data[file_list[2]][sensor_R],
                                    *data[file_list[2]][sensor_L], 
                                    ):
        whole_data = np.vstack((whole_data, np.asarray(same_timepoint_tuple)))
 
    return whole_data
