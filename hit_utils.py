import numpy as np
import scipy.signal as signal
import plistlib
from datetime import datetime

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
    """
    移动平均滤波器
    
    Args:
        data: 输入信号
        window_size: 窗口大小
        
    Returns:
        filtered_data: 经过移动平均滤波后的信号
    """
    window_size = int(window_size)
    half_window = window_size // 2

    padded_data = np.pad(data,
                         (half_window, half_window),
                         mode='edge')

    ma = np.cumsum(padded_data, dtype=float)
    ma[window_size:] = ma[window_size:] - ma[:-window_size]
    filtered_data = ma[window_size - 1:] / window_size

    return filtered_data[:len(data)]

def parse_new_hit_format(file_content):
    """
    解析新的统一HIT数据格式
    
    Args:
        file_content: plist文件内容
        
    Returns:
        dict: 包含所有解析数据的字典
    """
    try:
        plist_data = plistlib.loads(file_content)
        
        # 提取各个数据数组
        head_x = np.array(plist_data.get('HeadXList', []))
        head_y = np.array(plist_data.get('HeadYList', []))
        head_z = np.array(plist_data.get('HeadZList', []))
        
        left_eye_x = np.array(plist_data.get('LeftEyeXList', []))
        left_eye_y = np.array(plist_data.get('LeftEyeYList', []))
        
        right_eye_x = np.array(plist_data.get('RightEyeXList', []))
        right_eye_y = np.array(plist_data.get('RightEyeYList', []))
        right_eye_z = np.array(plist_data.get('RightEyeZList', []))
        
        time_strings = plist_data.get('TimeList', [])
        
        # 转换时间戳为相对时间
        timestamps = []
        if time_strings:
            reference_time = datetime.strptime(time_strings[0], '%Y-%m-%d %H:%M:%S.%f')
            for ts_str in time_strings:
                current_time = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
                relative_time = (current_time - reference_time).total_seconds()
                timestamps.append(relative_time)
        
        timestamps = np.array(timestamps)
        
        return {
            'head_x': head_x,
            'head_y': head_y,
            'head_z': head_z,
            'left_eye_x': left_eye_x,
            'left_eye_y': left_eye_y,
            'right_eye_x': right_eye_x,
            'right_eye_y': right_eye_y,
            'right_eye_z': right_eye_z,
            'timestamps': timestamps,
            'format': 'new_unified'
        }
        
    except Exception as e:
        raise ValueError(f"Error parsing new HIT format: {str(e)}")

def parse_old_hit_format(file_content, file_type):
    """
    解析旧的分离HIT数据格式
    
    Args:
        file_content: plist文件内容
        file_type: 文件类型 ('head', 'lefteye', 'righteye')
        
    Returns:
        np.array: 信号数据
    """
    try:
        plist_data = plistlib.loads(file_content)
        
        # 旧格式通常是简单的数组
        if isinstance(plist_data, list):
            return np.array(plist_data)
        else:
            raise ValueError("Old format should contain a simple array")
            
    except Exception as e:
        raise ValueError(f"Error parsing old HIT format: {str(e)}")

def signal_preprocess_hit(signal_data, timestamps, 
                         highpass_parameter={'cutoff': 0.1, 'fs': 60, 'order': 5}, 
                         lowpass_parameter={'cutoff': 6, 'fs': 60, 'order': 5}, 
                         window_size=25, interpolate_ratio=10):
    """
    HIT信号预处理
    
    Args:
        signal_data: 输入信号数据
        timestamps: 时间戳数组
        highpass_parameter: 高通滤波参数
        lowpass_parameter: 低通滤波参数
        window_size: 移动平均窗口大小
        interpolate_ratio: 插值比例
        
    Returns:
        tuple: (处理后的信号, 对应的时间数组)
    """
    if len(signal_data) == 0 or len(timestamps) == 0:
        return np.array([]), np.array([])
    
    # 确保数据长度一致
    min_len = min(len(signal_data), len(timestamps))
    signal_data = signal_data[:min_len]
    timestamps = timestamps[:min_len]
    
    freq = highpass_parameter['fs']
    signal_filtered = signal_data.copy()
    
    # 高通滤波
    signal_filtered = butter_highpass_filter(signal_filtered, **highpass_parameter)
    
    # 低通滤波
    signal_filtered = butter_lowpass_filter(signal_filtered, **lowpass_parameter)
    
    # 重采样
    signal_filtered = signal.resample(signal_filtered, int(len(signal_data) * interpolate_ratio))
    
    # 移动平均滤波
    if window_size > 0:
        signal_filtered = moving_average_filter(signal_filtered, window_size)
    
    # 生成新的时间序列
    time = np.linspace(timestamps[0], timestamps[-1], len(signal_filtered))
    
    return signal_filtered, time

def velocity_calculation_hit(signal_data, sampling_rate=600):
    """
    计算HIT信号的速度
    
    Args:
        signal_data: 输入信号
        sampling_rate: 采样率 (Hz)
        
    Returns:
        tuple: (速度信号, 时间序列)
    """
    velocity = np.diff(signal_data)
    velocity = velocity / (1 / sampling_rate)
    
    time = np.arange(len(velocity)) / sampling_rate
    
    return velocity, time

def velocity_peak_detection_hit(velocity, distance=200):
    """
    HIT速度峰值检测 - 分别检测正向和负向峰值
    
    Args:
        velocity: 速度信号
        distance: 峰值间最小距离
        
    Returns:
        dict: 包含正向和负向峰值的字典 {'positive_peaks': array, 'negative_peaks': array}
    """
    # 检测正向峰值（向上的峰值）
    positive_peaks, _ = signal.find_peaks(velocity, distance=distance)
    
    # 检测负向峰值（向下的峰值）
    negative_peaks, _ = signal.find_peaks(-velocity, distance=distance)
    
    # 过滤幅度较小的峰值
    if len(positive_peaks) > 0:
        positive_values = velocity[positive_peaks]
        positive_max = np.max(positive_values)
        positive_peaks = positive_peaks[positive_values > positive_max * 0.4]
    
    if len(negative_peaks) > 0:
        negative_values = velocity[negative_peaks]
        negative_min = np.min(negative_values)  # 注意这里是min因为是负值
        negative_peaks = negative_peaks[negative_values < negative_min * 0.4]
    
    return {
        'positive_peaks': positive_peaks,
        'negative_peaks': negative_peaks
    }

def velocity_get_zero(velocity):
    """
    获取速度过零点
    
    Args:
        velocity: 速度信号
        
    Returns:
        list: 过零点索引
    """
    zero_index = []
    for i in range(len(velocity) - 1):
        if velocity[i] < 0 and velocity[i+1] > 0:
            zero_index.append(i + 1)
        if velocity[i] > 0 and velocity[i+1] < 0:
            zero_index.append(i + 1)
    
    return zero_index

def vor_gain_calculation(head_velocity, eye_velocity, invert_signals=False):
    """
    计算VOR增益
    
    Args:
        head_velocity: 头部速度信号
        eye_velocity: 眼部速度信号
        invert_signals: 是否反转信号（用于处理负向峰值）
        
    Returns:
        tuple: (VOR增益, 分析信息字典)
    """
    try:
        # 如果需要反转信号（用于负向峰值）
        if invert_signals:
            head_velocity = -head_velocity
            eye_velocity = -eye_velocity
        
        head_peak_index = np.argmax(head_velocity)
        eye_peak_index = signal.find_peaks(eye_velocity)[0]
        
        if len(eye_peak_index) == 0:
            return np.nan, {}
        
        eye_peak_index = eye_peak_index[np.argmin(np.abs(eye_peak_index - head_peak_index))]
        
        head_zero_index = velocity_get_zero(head_velocity)
        eye_zero_index = velocity_get_zero(eye_velocity)
        
        head_left_zero_index = 0
        head_right_zero_index = len(head_velocity) - 1
        eye_left_zero_index = 0
        eye_right_zero_index = len(eye_velocity) - 1
        
        # 找到头部峰值周围的过零点
        for i in range(len(head_zero_index) - 1):
            if head_zero_index[i] < head_peak_index and head_zero_index[i+1] > head_peak_index:
                head_left_zero_index = head_zero_index[i]
                head_right_zero_index = head_zero_index[i+1]
                break
        
        # 找到眼部峰值周围的过零点
        for i in range(len(eye_zero_index) - 1):
            if eye_zero_index[i] < eye_peak_index and eye_zero_index[i+1] > eye_peak_index:
                eye_left_zero_index = eye_zero_index[i]
                eye_right_zero_index = eye_zero_index[i+1]
                break
        
        # 计算曲线下面积
        head_AUC = np.trapz(head_velocity[head_left_zero_index:head_right_zero_index])
        eye_AUC = np.trapz(eye_velocity[eye_left_zero_index:eye_right_zero_index])
        
        if head_AUC == 0:
            return np.nan, {}
        
        vor_gain = eye_AUC / head_AUC
        
        return vor_gain, {
            'head_indices': (head_left_zero_index, head_right_zero_index),
            'eye_indices': (eye_left_zero_index, eye_right_zero_index),
            'head_AUC': head_AUC,
            'eye_AUC': eye_AUC,
            'head_peak_index': head_peak_index,
            'eye_peak_index': eye_peak_index
        }
        
    except Exception as e:
        return np.nan, {'error': str(e)}

def analyze_hit_data(hit_data, eye_selection='Left Eye', 
                    highpass_params={'cutoff': 0.1, 'fs': 60, 'order': 5},
                    lowpass_params={'cutoff': 6, 'fs': 60, 'order': 5},
                    window_size=5, interpolate_ratio=10,
                    peak_distance=200, analysis_window_size=400):
    """
    完整的HIT数据分析流程
    
    Args:
        hit_data: 解析后的HIT数据字典
        eye_selection: 眼部选择 ('Left Eye', 'Right Eye', 'Both Eyes')
        其他参数: 各种分析参数
        
    Returns:
        dict: 分析结果
    """
    try:
        results = {}
        
        # 选择要分析的眼部数据
        if eye_selection == 'Left Eye' or eye_selection == 'Both Eyes':
            if len(hit_data['left_eye_x']) > 0:
                left_result = analyze_single_eye(
                    hit_data['head_x'], hit_data['left_eye_x'], hit_data['timestamps'],
                    'Left Eye', highpass_params, lowpass_params, window_size, 
                    interpolate_ratio, peak_distance, analysis_window_size
                )
                results['left_eye'] = left_result
        
        if eye_selection == 'Right Eye' or eye_selection == 'Both Eyes':
            if len(hit_data['right_eye_x']) > 0:
                right_result = analyze_single_eye(
                    hit_data['head_x'], hit_data['right_eye_x'], hit_data['timestamps'],
                    'Right Eye', highpass_params, lowpass_params, window_size,
                    interpolate_ratio, peak_distance, analysis_window_size
                )
                results['right_eye'] = right_result
        
        return results
        
    except Exception as e:
        return {'error': str(e)}

def analyze_single_eye(head_data, eye_data, timestamps, eye_name,
                      highpass_params, lowpass_params, window_size,
                      interpolate_ratio, peak_distance, analysis_window_size):
    """
    分析单个眼部的HIT数据
    
    Args:
        head_data: 头部运动数据
        eye_data: 眼部运动数据
        timestamps: 时间戳
        eye_name: 眼部名称
        其他: 分析参数
        
    Returns:
        dict: 单眼分析结果
    """
    try:
        # 信号预处理
        head_filtered, time_head = signal_preprocess_hit(
            head_data, timestamps, highpass_params, lowpass_params,
            window_size, interpolate_ratio
        )
        
        eye_filtered, time_eye = signal_preprocess_hit(
            eye_data, timestamps, highpass_params, lowpass_params,
            window_size, interpolate_ratio
        )
        
        # 速度计算
        head_velocity, time_head_velocity = velocity_calculation_hit(head_filtered)
        eye_velocity, time_eye_velocity = velocity_calculation_hit(eye_filtered)
        
        # 反转头部速度（与原HIT代码保持一致）
        head_velocity = -head_velocity
        
        # 峰值检测
        peaks_dict = velocity_peak_detection_hit(head_velocity, distance=peak_distance)
        
        # 分别分析正向和负向峰值
        results = {}
        
        # 分析正向峰值（向左转头）
        if len(peaks_dict['positive_peaks']) > 0:
            positive_analysis = analyze_peak_direction(
                head_velocity, eye_velocity, peaks_dict['positive_peaks'], 
                analysis_window_size, 'Leftward'
            )
            results['leftward'] = {
                **positive_analysis,
                'eye_name': f"{eye_name} (Leftward)",
                'direction': 'leftward',
                'head_data': head_filtered,
                'eye_data': eye_filtered,
                'time_data': time_head,
                'head_velocity': head_velocity,
                'eye_velocity': eye_velocity,
                'time_velocity': time_head_velocity,
            }
        
        # 分析负向峰值（向右转头）  
        if len(peaks_dict['negative_peaks']) > 0:
            negative_analysis = analyze_peak_direction(
                head_velocity, eye_velocity, peaks_dict['negative_peaks'], 
                analysis_window_size, 'Rightward'
            )
            results['rightward'] = {
                **negative_analysis,
                'eye_name': f"{eye_name} (Rightward)",
                'direction': 'rightward',
                'head_data': head_filtered,
                'eye_data': eye_filtered,
                'time_data': time_head,
                'head_velocity': head_velocity,
                'eye_velocity': eye_velocity,
                'time_velocity': time_head_velocity,
            }
        
        return results
        
    except Exception as e:
        return {'error': str(e)}

def analyze_peak_direction(head_velocity, eye_velocity, peaks, analysis_window_size, direction_name):
    """
    分析特定方向的峰值
    
    Args:
        head_velocity: 头部速度信号
        eye_velocity: 眼部速度信号  
        peaks: 峰值索引数组
        analysis_window_size: 分析窗口大小
        direction_name: 方向名称
        
    Returns:
        dict: 该方向的分析结果
    """
    vor_gains = []
    peak_analysis = []
    
    # 判断是否需要反转信号（对于右向/负向峰值）
    invert_signals = direction_name == 'Rightward'
    
    for peak_idx in peaks:
        # 定义分析窗口
        start_idx = max(0, peak_idx - analysis_window_size // 2)
        end_idx = min(len(head_velocity), peak_idx + analysis_window_size // 2)
        
        # 提取窗口内的速度数据
        head_velocity_segment = head_velocity[start_idx:end_idx]
        eye_velocity_segment = eye_velocity[start_idx:end_idx]
        
        # 计算VOR增益，对于右向峰值需要反转信号
        gain, analysis_info = vor_gain_calculation(head_velocity_segment, eye_velocity_segment, invert_signals=invert_signals)
        
        vor_gains.append(gain)
        peak_analysis.append({
            'peak_index': peak_idx,
            'window_start': start_idx,
            'window_end': end_idx,
            'gain': gain,
            'analysis_info': analysis_info
        })
    
    # 计算统计指标
    valid_gains = [g for g in vor_gains if not np.isnan(g)]
    
    if valid_gains:
        median_gain = np.median(valid_gains)
        std_gain = np.std(valid_gains)
        mean_gain = np.mean(valid_gains)
    else:
        median_gain = np.nan
        std_gain = np.nan
        mean_gain = np.nan
    
    return {
        'peaks': peaks,
        'vor_gains': vor_gains,
        'peak_analysis': peak_analysis,
        'median_gain': median_gain,
        'std_gain': std_gain,
        'mean_gain': mean_gain,
        'num_peaks': len(peaks),
        'num_valid_gains': len(valid_gains),
        'direction_name': direction_name
    } 