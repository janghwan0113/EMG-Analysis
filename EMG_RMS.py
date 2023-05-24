import numpy as np
import copy

def filtered_to_rms(filtered_data, window=0.025, overlap=0.0125):
    
    window = int(1000 * window)
    overlap = int(1000 * overlap)
    y_rms = []
    
    if len(filtered_data.shape) == 1:
        temp =[]
        for i in range((len(filtered_data)//overlap)-1):
            temp.append(np.sqrt(np.mean(np.square(filtered_data[overlap * i : (overlap * i) + window ]))))
        
        y_rms.extend(temp)
        
    else:
        
        for single_sensor in filtered_data:
            temp =[]
            for i in range((len(single_sensor)//overlap)-1):
                temp.append(np.sqrt(np.mean(np.square(single_sensor[overlap * i : (overlap * i) + window ]))))
            
            y_rms.append(temp)
        
    return np.array(y_rms)

def cropped_to_rms(cropped_emg_data, window=0.025, overlap=0.0125):
    window_length = int(window * 1000)
    overlap_length = int(overlap * 1000)

    for i in range(len(cropped_emg_data)):
        for j in range(len(cropped_emg_data[i])):
            temp = []
            for k in range(len(cropped_emg_data[i][j]) // overlap_length - 1):
                start = k * overlap_length
                end = start + window_length
                window_data = cropped_emg_data[i][j][start:end]
                rms = np.sqrt(np.mean(np.square(window_data)))
                temp.append(rms)
            cropped_emg_data[i][j] = temp

    return cropped_emg_data
