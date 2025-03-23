import numpy as np
import scipy.signal as signal
import plistlib

def butter_highpass_filter(data, cutoff, fs, order=5):
    """
    零相位高通巴特沃斯滤波器

    Args:
        data: 输入信号
        cutoff: 截止频率 (Hz)
        fs: 采样频率 (Hz)
        order: 滤波器阶数

    Returns:
        filtered_data: 经过零相位滤波后的信号
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # 确保截止频率在有效范围内
    if normal_cutoff >= 1:
        return data

    # 设计滤波器
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

    # 使用filtfilt进行双向滤波，消除相位延迟
    filtered_data = signal.filtfilt(b, a, data, padlen=min(len(data)-1, 3*(max(len(b), len(a))-1)))

    return filtered_data

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    零相位低通巴特沃斯滤波器

    Args:
        data: 输入信号
        cutoff: 截止频率 (Hz)
        fs: 采样频率 (Hz)
        order: 滤波器阶数

    Returns:
        filtered_data: 经过零相位滤波后的信号
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # 确保截止频率在有效范围内
    if normal_cutoff >= 1:
        return data

    # 设计滤波器
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # 使用filtfilt进行双向滤波，消除相位延迟
    filtered_data = signal.filtfilt(b, a, data, padlen=min(len(data)-1, 3*(max(len(b), len(a))-1)))

    return filtered_data
def moving_average_filter(data, window_size):

    window_size = int(window_size)
    half_window = window_size // 2

    padded_data = np.pad(data,
                         (half_window, half_window),
                         mode='edge')

    ma = np.cumsum(padded_data, dtype=float)
    ma[window_size:] = ma[window_size:] - ma[:-window_size]
    filtered_data = ma[window_size - 1:] / window_size

    return filtered_data[:len(data)]

def plists_to_data(plist_path):
    with open(plist_path, 'rb') as fp:
        plist_data = plistlib.load(fp)
    return plist_data


def signal_preprocess(plist_path, highpass_parameter={'cutoff': 0.1, 'fs': 60, 'order': 5}, lowpass_parameter={'cutoff': 6, 'fs': 60, 'order': 5}, window_size=25, interpolate_ratio=10):

    freq = highpass_parameter['fs']
    signal_data = plists_to_data(plist_path)
    signal_data_filtered = signal_data.copy()

    signal_data_filtered = butter_highpass_filter(signal_data_filtered, **highpass_parameter)
    signal_data_filtered = butter_lowpass_filter(signal_data_filtered, **lowpass_parameter)
    signal_data_filtered = signal.resample(signal_data_filtered, int(len(signal_data) * interpolate_ratio))
    signal_data_filtered = moving_average_filter(signal_data_filtered, window_size)

    time = np.arange(len(signal_data_filtered)) / (freq * interpolate_ratio)

    return signal_data_filtered, time


def velocity_calculation(signal_data, sampling_rate=600):

    # the sampling rate is 600 Hz
    velocity = np.diff(signal_data)
    velocity = velocity / (1 / sampling_rate)

    time = np.arange(len(velocity)) / sampling_rate

    return velocity, time


def velocity_peak_detection(velocity, distance=200):

    peak_index = signal.find_peaks(abs(velocity), distance=distance)

    peak_index = peak_index[0]
    peak_index = peak_index[velocity[peak_index] > 0]

    peak_value = velocity[peak_index]
    peak_max = np.max(peak_value)
    peak_index = peak_index[velocity[peak_index] > peak_max * 0.4]

    return peak_index


def VOR_gain_calculation(head_velocity, eye_velocity):
    
    head_peak_index = np.argmax(head_velocity)
    eye_peak_index = signal.find_peaks(eye_velocity)[0]
    eye_peak_index = eye_peak_index[np.argmin(np.abs(eye_peak_index - head_peak_index))]
    

    head_zero_index = velocity_get_zero(head_velocity)
    eye_zero_index = velocity_get_zero(eye_velocity)
    
    head_left_zero_index = 0
    head_right_zero_index = 0
    eye_left_zero_index = 0
    eye_right_zero_index = 0
    
    for i in range(len(head_zero_index) - 1):
        if head_zero_index[i] < head_peak_index and head_zero_index[i+1] > head_peak_index:
            head_left_zero_index = head_zero_index[i]
            head_right_zero_index = head_zero_index[i+1]
            break
        
    for i in range(len(eye_zero_index) - 1):
        if eye_zero_index[i] < eye_peak_index and eye_zero_index[i+1] > eye_peak_index:
            eye_left_zero_index = eye_zero_index[i]
            eye_right_zero_index = eye_zero_index[i+1]
            break
        
    head_AUC = np.trapz(head_velocity[head_left_zero_index:head_right_zero_index])
    lefteye_AUC = np.trapz(eye_velocity[eye_left_zero_index:eye_right_zero_index])
    
    VOR_gain = lefteye_AUC / head_AUC
    
    # 计算并返回更多信息用于可视化
    head_segment = head_velocity[head_left_zero_index:head_right_zero_index]
    eye_segment = eye_velocity[eye_left_zero_index:eye_right_zero_index]
    
    return VOR_gain, {
        'head_indices': (head_left_zero_index, head_right_zero_index),
        'eye_indices': (eye_left_zero_index, eye_right_zero_index),
        'head_AUC': head_AUC,
        'eye_AUC': lefteye_AUC
    }
    
    
    
def velocity_get_zero(velocity):
    
    # get the zero index
    # the zero index is the index of the velocity that is close to zero
    # velocity[i] < 0 and velocity[i+1] > 0
    # the i will be the zero index
    
    zero_index = []
    for i in range(len(velocity) - 1):
        if velocity[i] < 0 and velocity[i+1] > 0:
            zero_index.append(i + 1)
            
        if velocity[i] > 0 and velocity[i+1] < 0:
            zero_index.append(i + 1)

    return zero_index


if __name__ == "__main__":

    import os
    import matplotlib.pyplot as plt

    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'data')
    fig_path = os.path.join(base_path, 'fig')

    window_size = 15
    interpolate_ratio = 10
    peak_window = 300

    # for root, dirs, files in os.walk(data_path):
    #     for dic in dirs:

    #         title = f'{dic} Signal'
    #         fig = plt.figure(figsize=(14, 6))

    #         for file in os.listdir(os.path.join(root, dic)):
    #             if file.endswith('.plist'):
    #                 signal_data, time = signal_preprocess(os.path.join(root, dic, file), window_size=window_size, interpolate_ratio=interpolate_ratio)
    #                 velocity, time_velocity = velocity_calculation(signal_data)
    #                 if 'lefteye' in file:
    #                     plt.plot(time_velocity, velocity, label='lefteye')
    #                 elif 'head' in file:
    #                     velocity = -velocity
    #                     plt.plot(time_velocity, velocity, label='head')

    #                     peaks = velocity_peak_detection(velocity)
    #                     peaks = peaks
    #                     print(peaks)
    #                     plt.plot(time_velocity[peaks], velocity[peaks], 'ro')

    #         plt.title(title)
    #         plt.legend()
    #         plt.xlabel('Time (s)')
    #         plt.ylabel('Velocity (°/s)')

    #         plt.show()

    for root, dirs, files in os.walk(data_path):
        for dic in dirs:

            title = f'{dic} Signal'

            for file in os.listdir(os.path.join(root, dic)):

                head_file = f'{dic}_pHIT_head.plist'
                lefteye_file = f'{dic}_pHIT_lefteye.plist'

                head_signal_data, time_head = signal_preprocess(os.path.join(root, dic, head_file), window_size=window_size, interpolate_ratio=interpolate_ratio)
                head_velocity, time_velocity = velocity_calculation(head_signal_data)
                head_velocity = -head_velocity

                lefteye_signal_data, time_lefteye = signal_preprocess(os.path.join(root, dic, lefteye_file), window_size=window_size, interpolate_ratio=interpolate_ratio)
                lefteye_velocity, time_lefteye_velocity = velocity_calculation(lefteye_signal_data)

                peaks = velocity_peak_detection(head_velocity)
                
                for peak in peaks:
                    
                    start_index = peak - peak_window // 2
                    end_index = peak + peak_window // 2

                    head_velocity_segment = head_velocity[start_index:end_index]
                    lefteye_velocity_segment = lefteye_velocity[start_index:end_index]
                    
                    # 计算VOR增益和获取面积信息
                    gain, area_info = VOR_gain_calculation(head_velocity_segment, lefteye_velocity_segment)
                    
                    fig = plt.figure(figsize=(4, 4))
                    
                    # 绘制速度曲线
                    plt.plot(time_head[start_index:end_index], head_velocity_segment, label='head')
                    plt.plot(time_lefteye_velocity[start_index:end_index], lefteye_velocity_segment, label='lefteye')
                    
                    # 填充头部速度曲线下的面积
                    plt.fill_between(
                        time_head[start_index:end_index][area_info['head_indices'][0]:area_info['head_indices'][1]], 
                        head_velocity_segment[area_info['head_indices'][0]:area_info['head_indices'][1]], 
                        alpha=0.3, 
                        color='blue', 
                        label=f'Head AUC: {area_info["head_AUC"]:.2f}'
                    )
                    
                    # 填充眼睛速度曲线下的面积
                    plt.fill_between(
                        time_lefteye_velocity[start_index:end_index][area_info['eye_indices'][0]:area_info['eye_indices'][1]], 
                        lefteye_velocity_segment[area_info['eye_indices'][0]:area_info['eye_indices'][1]], 
                        alpha=0.3, 
                        color='orange', 
                        label=f'Eye AUC: {area_info["eye_AUC"]:.2f}'
                    )
                    
                    plt.title(f'{dic} Peak {peak}\nVOR Gain: {gain:.2f}')
                    plt.legend()
                    plt.xlabel('Time (s)')
                    plt.ylabel('Velocity (°/s)')
                    
                    plt.show()








