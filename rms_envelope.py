import numpy as np
import butter
import copy

def mov_avg(cropped_emg_rms):
    cropped_emg_rms_ = np.copy(cropped_emg_rms)

    # Calculate window size based on 5% of array length
    window_size = int(len(cropped_emg_rms_[0][0]) * 0.05)  

    for i in range(len(cropped_emg_rms_)):
        for j in range(len(cropped_emg_rms_[i])):
            cropped_emg_rms_[i][j] = np.convolve(cropped_emg_rms_[i][j], np.ones(window_size) / window_size, mode='valid')

    return cropped_emg_rms_

def freq_filter(cropped_emg_rms, cutoff, fs):

    cropped_emg_rms_ = copy.deepcopy(cropped_emg_rms)
    
    for i in range(len(cropped_emg_rms_)):
        for j in range(len(cropped_emg_rms_[i])):
            cropped_emg_rms_[i][j] = butter.butter_lowpass_filter(cropped_emg_rms_[i][j],cutoff, fs)
    
    return cropped_emg_rms_