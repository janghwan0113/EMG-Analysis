from scipy.signal import butter, lfilter, filtfilt
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    
    return b, a

def butter_bandstop(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    
    return b, a

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    
    return y

def butter_bandstop_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)    
    # y = lfilter(b, a, data)
    y = filtfilt(b, a, data)

    return y

def butter_lowpass_filter(data, cutoff, fs, order=4, Plot=False):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    
    return y

def filter_plot(filter):
    # For example :
    # filter = b, a = signal.butter(4, [100,200], 'band', analog=True)
    
    w, h = signal.freqs(filter)
    
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green') # cutoff frequency
    plt.show()