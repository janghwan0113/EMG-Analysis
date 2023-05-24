import numpy as np

def mean_std_out(reformed_data):
    #Reform data(101*60 numpy ndarray. 100 percent timepoint ROW. 60 action COLUMN.) to change to percentage
    reformed_data *= 100

    return np.mean(reformed_data, axis=1), np.std(reformed_data, axis=1)