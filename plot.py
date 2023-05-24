import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def mean_std_plot(mean_arr, std_arr, subplot=(1, 1, 1), title='Title', xlabel = 'X Label', ylabel = 'Y Label', legend='label'):
    
    #subplot = (row, column, index)
    x = np.linspace(0, 100, len(mean_arr))
    y = mean_arr
    cs = CubicSpline(x,y)
    
    plt.subplot(*subplot)    
    
    plt.plot(x, cs(x), label = legend)
    plt.fill_between(x, y - std_arr, y + std_arr, alpha=0.2)
    
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 80, 5))
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)