# import matlab.engine
import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, lfilter
import matplotlib.pyplot as plt
import copy
import time
import math

from butter import *
from EMG_RMS import filtered_to_rms, cropped_to_rms
from get_timepoint import *
from signal_slice import kinematic_crop
import rms_envelope
from find_max import max_of_sensor
from value_norm import mvc_norm
from time_norm import interpol_and_norm
from reform_data import data_reform
from plot import mean_std_plot
from mean_std import mean_std_out

#EMG, Kinematic csv file directory
dir_emg =       'C:/Users/Ahn/Desktop/OneDrive - SNU/Sports Engineering Lab/DATA/ANGEL/sub2_AJH_230511/Resampled'
dir_kinematic = 'C:/Users/Ahn/Desktop/OneDrive - SNU/Sports Engineering Lab/DATA/ANGEL/sub2_AJH_230511/box_kinematic'

#Read csv file and transform it into pandas dataframe
file_name = ['OFF_1','OFF_2','OFF_3','ON_1','ON_2','ON_3']
raw_data_dict  = {}

#EMG sensor attachment order
EMG_Sensor =   {'R_biceps':0,
                'R_anterior deltoid':1,
                'L_biceps':2,
                'L_anterior deltoid':3,
                'R_erector spinae muscle':4,
                'L_erector spinae muscle':5,
                'R_gluteus maximus':6,
                'L_gluteus maximus':7,
                'R_biceps femoris':8,
                'L_biceps femoris':9,
}

#Transform pandas data into numpy data in a dictionary.
for name in file_name: 
    data_frame = pd.read_csv(f'{dir_emg}/{name}.csv')
    
    # Each one row is data of one sensor.
    raw_data_dict[name] = data_frame.to_numpy().T[1::2]
    
    # Remove Offset
    raw_data_dict[name] -= np.mean(raw_data_dict[name], axis=1, keepdims=True)

#From Raw to enveloped RMS 
def emg_pipeline(data_dict):
    
    lift_enveloped = {}
    lower_enveloped = {}
        
    for trial, emg_data in data_dict.items():
        butter_bandpassed = butter_bandpass_filter(emg_data, lowcut=20, highcut=450, fs=1000, order=4)
        
        cropped_emg_lift, cropped_emg_lower = kinematic_crop(butter_bandpassed, get_peak(f'{dir_kinematic}/{trial}.csv'))
        
        cropped_emg_rms_lift, cropped_emg_rms_lower = cropped_to_rms(cropped_emg_lift), cropped_to_rms(cropped_emg_lower)
        
        rms_envelope_lift, rms_envelope_lower = rms_envelope.mov_avg(cropped_emg_rms_lift), rms_envelope.mov_avg(cropped_emg_rms_lower)
        # rms_envelope_lift, rms_envelope_lower = rms_envelope.freq_filter(cropped_emg_rms_lift,10,80), rms_envelope.freq_filter(cropped_emg_rms_lower,10,80)

        lift_enveloped[trial], lower_enveloped[trial] = rms_envelope_lift, rms_envelope_lower
    
    return lift_enveloped, lower_enveloped
    
if __name__ == '__main__':
    
    start = time.time()

    #RMS_enveloped
    lift_enveloped, lower_enveloped = emg_pipeline(raw_data_dict)
    
    #Maximun Value normalization 
    mvc_normed_lift = mvc_norm(lift_enveloped, max_of_sensor(lift_enveloped))
    mvc_normed_lower = mvc_norm(lower_enveloped, max_of_sensor(lower_enveloped))
    
    #Percent normalization  
    percent_normed_lift, percent_normed_lower = interpol_and_norm(mvc_normed_lift), interpol_and_norm(mvc_normed_lower)
    
    #Data reform preparation for mean, std calculation
    #data reform. 좌우, 같은 trial 안, 다른 trial action 같은 timepoint별로 구성. 100*60 array. 100=percent data 개수. 60은 10(1trial 10번)*2(좌우)*3(1,2,3 trial)
    ########################################### LIFT_OFF ##########################################
    biceps_lift_off_reformed_data = data_reform(percent_normed_lift, file_name[:3], EMG_Sensor['R_biceps'], EMG_Sensor['L_biceps'])
    deltoid_lift_off_reformed_data = data_reform(percent_normed_lift, file_name[:3], EMG_Sensor['R_anterior deltoid'], EMG_Sensor['L_anterior deltoid'])
    erector_lift_off_reformed_data = data_reform(percent_normed_lift, file_name[:3], EMG_Sensor['R_erector spinae muscle'], EMG_Sensor['L_erector spinae muscle'])
    gluteus_lift_off_reformed_data = data_reform(percent_normed_lift, file_name[:3], EMG_Sensor['R_gluteus maximus'], EMG_Sensor['R_gluteus maximus'])
    femoris_lift_off_reformed_data = data_reform(percent_normed_lift, file_name[:3], EMG_Sensor['R_biceps femoris'], EMG_Sensor['L_biceps femoris'])
    
    ########################################### LIFT_ON ###########################################
    biceps_lift_on_reformed_data = data_reform(percent_normed_lift, file_name[3:], EMG_Sensor['R_biceps'], EMG_Sensor['L_biceps'])
    deltoid_lift_on_reformed_data = data_reform(percent_normed_lift, file_name[3:], EMG_Sensor['R_anterior deltoid'], EMG_Sensor['L_anterior deltoid'])
    erector_lift_on_reformed_data = data_reform(percent_normed_lift, file_name[3:], EMG_Sensor['R_erector spinae muscle'], EMG_Sensor['L_erector spinae muscle'])
    gluteus_lift_on_reformed_data = data_reform(percent_normed_lift, file_name[3:], EMG_Sensor['R_gluteus maximus'], EMG_Sensor['R_gluteus maximus'])
    femoris_lift_on_reformed_data = data_reform(percent_normed_lift, file_name[3:], EMG_Sensor['R_biceps femoris'], EMG_Sensor['L_biceps femoris'])
    
    ########################################### LOWER_OFF #########################################
    biceps_lower_off_reformed_data = data_reform(percent_normed_lower, file_name[:3], EMG_Sensor['R_biceps'], EMG_Sensor['L_biceps'])
    deltoid_lower_off_reformed_data = data_reform(percent_normed_lower, file_name[:3], EMG_Sensor['R_anterior deltoid'], EMG_Sensor['L_anterior deltoid'])
    erector_lower_off_reformed_data = data_reform(percent_normed_lower, file_name[:3], EMG_Sensor['R_erector spinae muscle'], EMG_Sensor['L_erector spinae muscle'])
    gluteus_lower_off_reformed_data = data_reform(percent_normed_lower, file_name[:3], EMG_Sensor['R_gluteus maximus'], EMG_Sensor['R_gluteus maximus'])
    femoris_lower_off_reformed_data = data_reform(percent_normed_lower, file_name[:3], EMG_Sensor['R_biceps femoris'], EMG_Sensor['L_biceps femoris'])
    
    ########################################### LOWER_ON ###########################################
    biceps_lower_on_reformed_data = data_reform(percent_normed_lower, file_name[3:], EMG_Sensor['R_biceps'], EMG_Sensor['L_biceps'])
    deltoid_lower_on_reformed_data = data_reform(percent_normed_lower, file_name[3:], EMG_Sensor['R_anterior deltoid'], EMG_Sensor['L_anterior deltoid'])
    erector_lower_on_reformed_data = data_reform(percent_normed_lower, file_name[3:], EMG_Sensor['R_erector spinae muscle'], EMG_Sensor['L_erector spinae muscle'])
    gluteus_lower_on_reformed_data = data_reform(percent_normed_lower, file_name[3:], EMG_Sensor['R_gluteus maximus'], EMG_Sensor['R_gluteus maximus'])
    femoris_lower_on_reformed_data = data_reform(percent_normed_lower, file_name[3:], EMG_Sensor['R_biceps femoris'], EMG_Sensor['L_biceps femoris'])
    
    
    #################################################### MEAN STD PLOT #######################################################
    
    ###############################
    ########## L I F T ############
    ###############################
    
    ########### LIFT BICEPS ON OFF #########
    biceps_lift_off_mean_arr, biceps_lift_off_std_arr = mean_std_out(biceps_lift_off_reformed_data)
    
    mean_std_plot(biceps_lift_off_mean_arr, biceps_lift_off_std_arr,(2,5,1),'Biceps_Lift', 'Lifting (%)','Maximum Value (%)', 'OFF')

    biceps_lift_on_mean_arr, biceps_lift_on_std_arr = mean_std_out(biceps_lift_on_reformed_data)
    mean_std_plot(biceps_lift_on_mean_arr, biceps_lift_on_std_arr, (2,5,1),'Biceps_Lift', 'Lifting (%)','Maximum Value (%)', 'ON')
    
    ########## LIFT DELTOID ON OFF #########
    deltoid_lift_off_mean_arr, deltoid_lift_off_std_arr = mean_std_out(deltoid_lift_off_reformed_data)
    mean_std_plot(deltoid_lift_off_mean_arr, deltoid_lift_off_std_arr, (2,5,2),'Deltoid_Lift', 'Lifting (%)','Maximum Value (%)', 'OFF')

    deltoid_lift_on_mean_arr, deltoid_lift_on_std_arr = mean_std_out(deltoid_lift_on_reformed_data)
    mean_std_plot(deltoid_lift_on_mean_arr, deltoid_lift_on_std_arr, (2,5,2),'Deltoid_Lift', 'Lifting (%)','Maximum Value (%)', 'ON')
    
    ########### LIFT ERECTOR ON OFF #########
    erector_lift_off_mean_arr, erector_lift_off_std_arr = mean_std_out(erector_lift_off_reformed_data)
    
    mean_std_plot(erector_lift_off_mean_arr, erector_lift_off_std_arr, (2,5,3),'Erector_Lift', 'Lifting (%)','Maximum Value (%)', 'OFF')

    erector_lift_on_mean_arr, erector_lift_on_std_arr = mean_std_out(erector_lift_on_reformed_data)
    mean_std_plot(erector_lift_on_mean_arr, erector_lift_on_std_arr, (2,5,3),'Erector_Lift', 'Lifting (%)','Maximum Value (%)', 'ON')
    
    ########## LIFT GLUTEUS ON OFF #########
    gluteus_lift_off_mean_arr, gluteus_lift_off_std_arr = mean_std_out(gluteus_lift_off_reformed_data)
    
    mean_std_plot(gluteus_lift_off_mean_arr, gluteus_lift_off_std_arr, (2,5,4),'Gluteus_Lift', 'Lifting (%)','Maximum Value (%)', 'OFF')

    gluteus_lift_on_mean_arr, gluteus_lift_on_std_arr = mean_std_out(gluteus_lift_on_reformed_data)
    mean_std_plot(gluteus_lift_on_mean_arr, gluteus_lift_on_std_arr, (2,5,4),'Gluteus_Lift', 'Lifting (%)','Maximum Value (%)', 'ON')
    
    ########### LIFT FEMORIS ON OFF #########
    femoris_lift_off_mean_arr, femoris_lift_off_std_arr = mean_std_out(femoris_lift_off_reformed_data)
    
    mean_std_plot(femoris_lift_off_mean_arr, femoris_lift_off_std_arr, (2,5,5),'Femoris_Lift', 'Lifting (%)','Maximum Value (%)', 'OFF')

    femoris_lift_on_mean_arr, femoris_lift_on_std_arr = mean_std_out(femoris_lift_on_reformed_data)
    mean_std_plot(femoris_lift_on_mean_arr, femoris_lift_on_std_arr, (2,5,5),'Femoris_Lift', 'Lifting (%)','Maximum Value (%)', 'ON')
    
    ###############################
    ########### L O W E R  ########
    ###############################
    
    ######### LOWER BICEPS ON OFF #########
    biceps_lower_off_mean_arr, biceps_lower_off_std_arr = mean_std_out(biceps_lower_off_reformed_data)
    
    mean_std_plot(biceps_lower_off_mean_arr, biceps_lower_off_std_arr, (2,5,6),'Biceps_Lower', 'Lowering (%)','Maximum Value (%)', 'OFF')

    biceps_lower_on_mean_arr, biceps_lower_on_std_arr = mean_std_out(biceps_lower_on_reformed_data)
    mean_std_plot(biceps_lower_on_mean_arr, biceps_lower_on_std_arr, (2,5,6),'Biceps_Lower', 'Lowering (%)','Maximum Value (%)', 'ON')

    ########## LOWER DELTOID ON OFF #########
    deltoid_lower_off_mean_arr, deltoid_lower_off_std_arr = mean_std_out(deltoid_lower_off_reformed_data)
    
    mean_std_plot(deltoid_lower_off_mean_arr, deltoid_lower_off_std_arr, (2,5,7),'Deltoid_Lower', 'Lowering (%)','Maximum Value (%)', 'OFF')

    deltoid_lower_on_mean_arr, deltoid_lower_on_std_arr = mean_std_out(deltoid_lower_on_reformed_data)
    mean_std_plot(deltoid_lower_on_mean_arr, deltoid_lower_on_std_arr, (2,5,7),'Deltoid_Lower', 'Lowering (%)','Maximum Value (%)', 'ON')
    
    ########### LOWER ERECTOR ON OFF #########
    erector_lower_off_mean_arr, erector_lower_off_std_arr = mean_std_out(erector_lower_off_reformed_data)
    
    mean_std_plot(erector_lower_off_mean_arr, erector_lower_off_std_arr, (2,5,8),'Erector_Lower', 'Lowering (%)','Maximum Value (%)', 'OFF')

    erector_lower_on_mean_arr, erector_lower_on_std_arr = mean_std_out(erector_lower_on_reformed_data)
    mean_std_plot(erector_lower_on_mean_arr, erector_lower_on_std_arr, (2,5,8),'Erector_Lower', 'Lowering (%)','Maximum Value (%)', 'ON')
    
    ########## LOWER GLUTEUS ON OFF #########
    gluteus_lower_off_mean_arr, gluteus_lower_off_std_arr = mean_std_out(gluteus_lower_off_reformed_data)
    
    mean_std_plot(gluteus_lower_off_mean_arr, gluteus_lower_off_std_arr, (2,5,9),'Gluteus_Lower', 'Lowering (%)','Maximum Value (%)', 'OFF')

    gluteus_lower_on_mean_arr, gluteus_lower_on_std_arr = mean_std_out(gluteus_lower_on_reformed_data)
    mean_std_plot(gluteus_lower_on_mean_arr, gluteus_lower_on_std_arr, (2,5,9),'Gluteus_Lower', 'Lowering (%)','Maximum Value (%)', 'ON')
    
    ########## LOWER FEMORIS ON OFF #########
    femoris_lower_off_mean_arr, femoris_lower_off_std_arr = mean_std_out(femoris_lower_off_reformed_data)
    
    mean_std_plot(femoris_lower_off_mean_arr, femoris_lower_off_std_arr, (2,5,10),'Femoris_Lower', 'Lowering (%)','Maximum Value (%)', 'OFF')

    femoris_lower_on_mean_arr, femoris_lower_on_std_arr = mean_std_out(femoris_lower_on_reformed_data)
    mean_std_plot(femoris_lower_on_mean_arr, femoris_lower_on_std_arr, (2,5,10),'Femoris_Lower', 'Lowering (%)','Maximum Value (%)', 'ON')
    
    end = time.time()
    print(f"exection time : {end - start:.5f} sec")
    
    plt.show()