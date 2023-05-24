import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from butter import *

def get_peak(dir_csv):

    try :
        # CSV 파일 불러오기
        data = pd.read_csv(dir_csv)
    
    except :
        print("Check file format. File format must be 'csv'")

    else : 
        # 데이터 프레임에서 필요한 열 선택
        x = data['Frame']
        y = data['R Z']

        # y 값에 저주파 필터링
        b, a = butter_lowpass(3, 100, order=4)
        y_filtered = lfilter(b, a, y)

        # 큰 피크와 작은 피크를 모두 찾기
        peaks_small, _ = find_peaks(-y_filtered, height=-500)
        peaks_large, _ = find_peaks(y_filtered, height=1040)

        # 피크를 리스트로 합치기
        peaks_small = list(peaks_small)
        peaks_large = list(peaks_large)


        # 시작과 끝 지점을 저장할 리스트 생성
        lift_start = []
        lift_end = []
        lower_end = []
        lower_start = []

        # 피크 그룹 시작점과 끝점 찾기
        for i in range(len(peaks_small)-1):
            if peaks_small[i+1] - peaks_small[i] > 500:  # 100 프레임 이상 차이나는 경우에만 피크 그룹으로 판단
                lift_start.append(peaks_small[i-1])
                lower_end.append(peaks_small[i+1]-1)  # 끝점은 다음 피크 시작점 - 1

        lift_end.append(peaks_large[1]) #시작점

        for i in range(len(peaks_large)-1):
            if peaks_large[i+1] - peaks_large[i] > 500:  # 100 프레임 이상 차이나는 경우에만 피크 그룹으로 판단
                lift_end.append(peaks_large[i + 2]) #시작점에서 다음을 피크로 지정
                lower_start.append(peaks_large[i-3]-1)  # 끝점은 다음 피크 시작점 - 1

        lower_start.append(peaks_large[len(peaks_large)-4]) #끝점

        #Multiply 10 for every element. Motion Capture system 100Hz, EMG 1000Hz lowersampled.
        lift_start = [x * 10 for x in lift_start]
        lift_end = [x * 10 for x in lift_end]
        lower_start = [x * 10 for x in lower_start]
        lower_end = [x * 10 for x in lower_end]

        return (lift_start, lift_end, lower_start, lower_end)
    

def get_peak_plot(dir_csv):
    
    try:
        # CSV 파일 불러오기
        data = pd.read_csv(dir_csv)

    except :
        print("Check file format. File format must be 'csv'")

    else :
        # 데이터 프레임에서 필요한 열 선택
        x = data['Frame']
        y = data['R Z']

        # y 값에 저주
        # 
        # 
        # 파 필터링
        b, a = butter_lowpass(3, 100, order=4)
        y_filtered = lfilter(b, a, y)
        
        # 큰 피크와 작은 피크를 모두 찾기
        peaks_small, _ = find_peaks(-y_filtered, height = -500)
        peaks_large, _ = find_peaks(y_filtered, height = 1040)

        # 피크를 리스트로 합치기
        peaks_small = list(peaks_small)
        peaks_large = list(peaks_large)


        # 시작과 끝 지점을 저장할 리스트 생성
        lift_start = []
        lift_end = []
        lower_end = []
        lower_start = []

        # 피크 그룹 시작점과 끝점 찾기
        for i in range(len(peaks_small)-1):
            if peaks_small[i+1] - peaks_small[i] > 500:  # 100 프레임 이상 차이나는 경우에만 피크 그룹으로 판단
                lift_start.append(peaks_small[i])
                lower_end.append(peaks_small[i+1]-1)  # 끝점은 다음 피크 시작점 - 1

        lift_end.append(peaks_large[1]) #시작점

        for i in range(len(peaks_large)-1):
            if peaks_large[i+1] - peaks_large[i] > 500:  # 100 프레임 이상 차이나는 경우에만 피크 그룹으로 판단
                lift_end.append(peaks_large[i + 2]) #시작점에서 다음을 피크로 지정
                lower_start.append(peaks_large[i-1]-1)  # 끝점은 다음 피크 시작점 - 1

        lower_start.append(peaks_large[len(peaks_large)-2]) #끝점
        
        # 시작점과 끝점에 파란색 'x' 표시
        plt.plot(x, y_filtered)
        for start, end in zip(lift_start, lower_end):
            plt.plot(x[start], y_filtered[start], 'bo')
            plt.plot(x[end], y_filtered[end], 'bo')

        plt.plot(x, y_filtered)
        for start, end in zip(lift_end, lower_start):
            plt.plot(x[start], y_filtered[start], 'bo')
            plt.plot(x[end], y_filtered[end], 'bo')

        # 모든 피크에 빨간색 'x' 표시
        for peak in peaks_small:
            plt.plot(x[peak], y_filtered[peak], 'rx')

        for peak in peaks_large:
            plt.plot(x[peak], y_filtered[peak], 'rx')

        plt.show()

