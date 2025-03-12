import numpy as np
import scipy.signal as signal
import plistlib
from datetime import datetime
from scipy.signal import find_peaks

def plists_to_data(plist_path):
    """
    从plist文件中读取数据并处理时间戳和眼球角度数据
    
    Args:
        plist_path: plist文件路径
        
    Returns:
        timestamps: 处理后的时间戳（浮点数格式，单位：秒）
        eye_angles: 眼球角度数据
    """
    with open(plist_path, 'rb') as fp:
        plist_data = plistlib.load(fp)
    
    # 处理时间戳
    timestamps_str = np.array(plist_data['TimeList'])
    timestamps = np.zeros(len(timestamps_str))
    
    # 选择参考时间点
    reference_time = datetime.strptime(timestamps_str[0], '%Y-%m-%d %H:%M:%S.%f')
    
    # 转换为相对时间（秒）
    for i, ts_str in enumerate(timestamps_str):
        current_time = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
        time_diff = (current_time - reference_time).total_seconds()
        timestamps[i] = time_diff
    
    # 获取眼球角度数据
    if 'lefteyeList' in plist_data:
        eye_angles = np.array(plist_data['lefteyeList'])
    elif 'righteyeList' in plist_data:
        eye_angles = np.array(plist_data['righteyeList'])
    
    return timestamps, eye_angles

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

def signal_preprocess(plist_path, highpass_parameter={'cutoff': 0.1, 'fs': 60, 'order': 5}, 
                     lowpass_parameter={'cutoff': 6, 'fs': 60, 'order': 5}, 
                     window_size=25, interpolate_ratio=10, skip_moving_average=False):
    """
    信号预处理完整流程
    
    Args:
        plist_path: plist文件路径
        highpass_parameter: 高通滤波器参数
        lowpass_parameter: 低通滤波器参数
        window_size: 移动平均窗口大小
        interpolate_ratio: 插值比例
        skip_moving_average: 是否跳过移动平均平滑
        
    Returns:
        signal_data_filtered: 处理后的信号
        time: 对应的时间序列
    """
    freq = highpass_parameter['fs']
    timestamps, eye_angles = plists_to_data(plist_path)
    
    # 应用高通滤波
    signal_filtered = butter_highpass_filter(eye_angles, **highpass_parameter)
    
    # 应用低通滤波
    signal_filtered = butter_lowpass_filter(signal_filtered, **lowpass_parameter)
    
    # 重采样（增加采样点）
    signal_filtered = signal.resample(signal_filtered, int(len(eye_angles) * interpolate_ratio))
    
    # 移动平均平滑 (只有当skip_moving_average为False时才执行)
    if not skip_moving_average and window_size > 0:
        signal_filtered = moving_average_filter(signal_filtered, window_size)
    
    # 生成新的时间序列
    time = np.linspace(timestamps[0], timestamps[-1], len(signal_filtered))
    
    return signal_filtered, time

def velocity_calculation(signal_data, time=None, sampling_rate=600):
    """
    计算信号的速度
    
    Args:
        signal_data: 输入信号
        time: 时间序列（如果提供，则使用不均匀差分）
        sampling_rate: 采样率 (Hz)
        
    Returns:
        velocity: 速度信号
        time_velocity: 对应的时间序列
    """
    if time is not None and len(time) == len(signal_data):
        # 使用不均匀时间间隔计算导数
        velocity = np.diff(signal_data) / np.diff(time)
        time_velocity = time[:-1] + np.diff(time)/2  # 速度点位于相邻时间点的中间
    else:
        # 使用固定采样率
        velocity = np.diff(signal_data) * sampling_rate
        time_velocity = np.arange(len(velocity)) / sampling_rate
    
    return velocity, time_velocity

# 新增函数: 拐点检测
def find_turning_points(signal_data, prominence=0.1, distance=100):
    """
    检测信号的拐点（局部极大值和极小值点）
    
    Args:
        signal_data: 输入信号
        prominence: 峰值突出度阈值
        distance: 相邻峰值之间的最小距离
    
    Returns:
        turning_points: 拐点索引列表
    """
    # 检测局部最大值
    peaks, _ = find_peaks(signal_data, prominence=prominence, distance=distance)
    # 检测局部最小值
    valleys, _ = find_peaks(-signal_data, prominence=prominence, distance=distance)
    
    # 合并并排序所有拐点
    turning_points = np.sort(np.concatenate([peaks, valleys]))
    
    return turning_points

# 新增函数: 计算斜率
def calculate_slopes(time, signal, turning_points):
    """
    计算相邻拐点之间的斜率
    
    Args:
        time: 时间序列
        signal: 信号数据
        turning_points: 拐点索引列表
    
    Returns:
        slope_times: 斜率对应的时间点（取两个拐点的中点）
        slopes: 斜率值列表
    """
    slopes = []
    slope_times = []
    
    for i in range(len(turning_points)-1):
        t1, t2 = time[turning_points[i]], time[turning_points[i+1]]
        y1, y2 = signal[turning_points[i]], signal[turning_points[i+1]]
        
        # 计算斜率 (度/秒)
        slope = (y2 - y1) / (t2 - t1)
        
        # 取两个拐点的中点作为该段斜率的时间点
        slope_time = (t1 + t2) / 2
        
        slopes.append(slope)
        slope_times.append(slope_time)
    
    return np.array(slope_times), np.array(slopes)

# 新增函数: 眼震模式分析
def analyze_nystagmus(slopes, times, slow_threshold=10, fast_threshold=20):
    """
    分析眼震的快相和慢相，并过滤掉与主要方向相反的检测结果
    
    Args:
        slopes: 斜率数组
        times: 斜率对应的时间点
        slow_threshold: 慢相速度阈值
        fast_threshold: 快相速度阈值
        
    Returns:
        spv: 平均慢相速度
        pattern: 'slow-fast' 或 'fast-slow'
        slow_phase_data: 慢相的时间和速度数据
        fast_phase_data: 快相的时间和速度数据
    """
    # 初始化存储慢相和快相数据的列表
    slow_phase_times = []
    slow_phase_velocities = []
    fast_phase_times = []
    fast_phase_velocities = []
    
    # 分析连续的三个斜率来识别眼震模式
    for i in range(1, len(slopes)-1):
        prev_slope = slopes[i-1]
        curr_slope = slopes[i]
        next_slope = slopes[i+1]
        
        # 判断当前段是否为慢相或快相
        if abs(curr_slope) < slow_threshold:  # 当前速度在慢相阈值范围内
            if (abs(prev_slope) > fast_threshold or abs(next_slope) > fast_threshold):  # 相邻有快相
                slow_phase_times.append(times[i])
                slow_phase_velocities.append(curr_slope)
        elif abs(curr_slope) > fast_threshold:  # 当前速度超过快相阈值
            fast_phase_times.append(times[i])
            fast_phase_velocities.append(curr_slope)
    
    if not slow_phase_velocities:
        return None, None, ([], []), ([], [])
    
    # 确定主要方向
    mean_velocity = np.mean(slow_phase_velocities)
    
    # 过滤掉与主要方向相反的检测结果
    filtered_slow_times = []
    filtered_slow_velocities = []
    filtered_fast_times = []
    filtered_fast_velocities = []
    
    if mean_velocity > 0:  # 如果主要方向是正向
        pattern = 'slow-fast (Left)'
        for t, v in zip(slow_phase_times, slow_phase_velocities):
            if v > 0:  # 只保留正向的慢相
                filtered_slow_times.append(t)
                filtered_slow_velocities.append(v)
        for t, v in zip(fast_phase_times, fast_phase_velocities):
            if v < 0:  # 保留负向的快相
                filtered_fast_times.append(t)
                filtered_fast_velocities.append(v)
    else:  # 如果主要方向是负向
        pattern = 'fast-slow (Right)'
        for t, v in zip(slow_phase_times, slow_phase_velocities):
            if v < 0:  # 只保留负向的慢相
                filtered_slow_times.append(t)
                filtered_slow_velocities.append(v)
        for t, v in zip(fast_phase_times, fast_phase_velocities):
            if v > 0:  # 保留正向的快相
                filtered_fast_times.append(t)
                filtered_fast_velocities.append(v)
    
    # 使用过滤后的数据计算最终的SPV
    spv = np.mean(filtered_slow_velocities) if filtered_slow_velocities else None
    
    return spv, pattern, (filtered_slow_times, filtered_slow_velocities), (filtered_fast_times, filtered_fast_velocities)

# 新增函数: 眼震模式识别
def identify_nystagmus_patterns(signal_data, time_data, min_time=0.3, max_time=0.8, min_ratio=1.4, max_ratio=8.0, direction_axis="horizontal"):
    """
    识别眼震模式
    
    Args:
        signal_data: 信号数据
        time_data: 时间序列
        min_time: 最小时间阈值（秒）
        max_time: 最大时间阈值（秒）
        min_ratio: 最小斜率比例
        max_ratio: 最大斜率比例
        direction_axis: 方向轴 ("horizontal" 或 "vertical")
        
    Returns:
        patterns: 识别出的眼震模式列表
        filtered_patterns: 被过滤掉的模式列表
        direction: 眼震方向
        spv: 慢相速度
        cv: 变异系数
    """
    # 检测拐点
    turning_points = find_turning_points(signal_data, prominence=0.1, distance=150)
    
    # 收集所有可能的眼震模式
    potential_patterns = []
    
    for i in range(1, len(turning_points)-1):
        idx1 = turning_points[i-1]
        idx2 = turning_points[i]
        idx3 = turning_points[i+1]
        
        p1 = np.array([time_data[idx1], signal_data[idx1]])
        p2 = np.array([time_data[idx2], signal_data[idx2]])
        p3 = np.array([time_data[idx3], signal_data[idx3]])
        
        # 计算总时长
        total_time = p3[0] - p1[0]
        
        # 检查时间阈值
        if not (min_time <= total_time <= max_time):
            continue
        
        # 计算斜率
        slope_before = (p2[1] - p1[1]) / (p2[0] - p1[0])
        slope_after = (p3[1] - p2[1]) / (p3[0] - p2[0])
        
        # 判断快慢相 - 使用绝对值比较斜率大小
        if abs(slope_before) > abs(slope_after):
            fast_slope = slope_before
            slow_slope = slope_after
            fast_phase_first = True
        else:
            fast_slope = slope_after
            slow_slope = slope_before
            fast_phase_first = False
        
        # 确保斜率符号一致性 - 快相应该与慢相方向相反
        # 如果快相和慢相斜率符号相同，这可能不是眼震模式
        if (fast_slope * slow_slope > 0):
            continue
            
        ratio = abs(fast_slope) / abs(slow_slope)
        
        if min_ratio <= ratio <= max_ratio:
            potential_patterns.append({
                'index': i,
                'time_point': time_data[idx2],
                'slow_slope': slow_slope,
                'fast_slope': fast_slope,
                'ratio': ratio,
                'fast_phase_first': fast_phase_first,
                'total_time': total_time
            })
    
    # 如果找到了潜在的眼震模式，进行CV筛选
    if potential_patterns:
        # 提取所有慢相斜率
        slow_slopes = np.array([p['slow_slope'] for p in potential_patterns])
        original_indices = list(range(len(potential_patterns)))
        
        # 计算初始变种CV
        median_slope = np.median(slow_slopes)
        mad = np.median(np.abs(slow_slopes - median_slope))
        mad_normalized = 1.4826 * mad
        cv = (mad_normalized / abs(median_slope)) * 100 if median_slope != 0 else float('inf')
        
        # 异常值过滤
        filtered_indices = []
        while cv > 20 and len(slow_slopes) > 3:
            modified_z_scores = 0.6745 * np.abs(slow_slopes - median_slope) / mad
            max_z_idx = np.argmax(modified_z_scores)
            filtered_indices.append(original_indices[max_z_idx])
            slow_slopes = np.delete(slow_slopes, max_z_idx)
            original_indices.pop(max_z_idx)
            
            median_slope = np.median(slow_slopes)
            mad = np.median(np.abs(slow_slopes - median_slope))
            mad_normalized = 1.4826 * mad
            cv = (mad_normalized / abs(median_slope)) * 100 if median_slope != 0 else float('inf')
        
        # 分配模式到保留和过滤两个列表
        patterns = []
        filtered_patterns = []
        
        for idx, pattern in enumerate(potential_patterns):
            if idx in filtered_indices:
                filtered_patterns.append(pattern)
            else:
                patterns.append(pattern)
        
        # 计算最终的方向和SPV
        final_median_slope = np.median(slow_slopes)
        
        # 根据direction_axis确定方向
        if direction_axis == "horizontal":
            direction = "left" if final_median_slope > 0 else "right"
        else:  # vertical
            direction = "up" if final_median_slope > 0 else "down"
            
        # 斜率计算时已经是度/秒单位，不需要再乘以60
        spv = abs(final_median_slope)  # 单位:度/秒
        
        return patterns, filtered_patterns, direction, spv, cv
    
    return [], [], "unknown", 0, float('inf')

# 新增函数: 完整的眼震分析流程
def analyze_eye_movement(plist_path, plot=False, save_path=None):
    """
    完整的眼震分析流程
    
    Args:
        plist_path: plist文件路径
        plot: 是否绘制图形
        save_path: 图形保存路径（如果为None则不保存）
        
    Returns:
        results: 包含分析结果的字典
    """
    # 预处理参数
    highpass_parameter = {'cutoff': 0.1, 'fs': 60, 'order': 5}
    lowpass_parameter = {'cutoff': 6, 'fs': 60, 'order': 5}
    window_size = 15
    interpolate_ratio = 10
    
    # 信号预处理
    filtered_signal, time = signal_preprocess(
        plist_path, 
        highpass_parameter=highpass_parameter,
        lowpass_parameter=lowpass_parameter,
        window_size=window_size,
        interpolate_ratio=interpolate_ratio
    )
    
    # 检测拐点
    turning_points = find_turning_points(filtered_signal)
    
    # 计算斜率
    slope_times, slopes = calculate_slopes(time, filtered_signal, turning_points)
    
    # 分析眼震特征
    spv, pattern, (slow_times, slow_velocities), (fast_times, fast_velocities) = analyze_nystagmus(slopes, slope_times)
    
    # 识别眼震模式
    patterns, filtered_patterns, direction, pattern_spv, cv = identify_nystagmus_patterns(
        filtered_signal, time
    )
    
    # 结果字典
    results = {
        'signal': filtered_signal,
        'time': time,
        'turning_points': turning_points,
        'slopes': slopes,
        'slope_times': slope_times,
        'spv': spv,
        'pattern': pattern,
        'slow_phase': (slow_times, slow_velocities),
        'fast_phase': (fast_times, fast_velocities),
        'nystagmus_patterns': patterns,
        'filtered_patterns': filtered_patterns,
        'direction': direction,
        'pattern_spv': pattern_spv,
        'cv': cv
    }
    
    # 绘图
    if plot:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # 绘制原始信号和拐点
        ax1.plot(time, filtered_signal, 'gray', alpha=0.5, label='Signal')
        ax1.plot(time[turning_points], filtered_signal[turning_points], 'ro', label='Turning Points', markersize=4)
        ax1.set_title('Eye Movement Signal with Turning Points')
        ax1.set_ylabel('Position (°)')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制眼震模式
        ax2.plot(time, filtered_signal, 'gray', alpha=0.5, label='Signal')
        
        # 先绘制被过滤掉的模式（浅色）
        for pattern in filtered_patterns:
            i = pattern['index']
            if i+2 < len(turning_points):
                idx1 = turning_points[i-1]
                idx2 = turning_points[i]
                idx3 = turning_points[i+1]
                
                if pattern['fast_phase_first']:
                    ax2.plot(time[idx1:idx2+1], filtered_signal[idx1:idx2+1],
                            'lightcoral', linewidth=2, alpha=0.5)
                    ax2.plot(time[idx2:idx3+1], filtered_signal[idx2:idx3+1],
                            'lightblue', linewidth=2, alpha=0.5)
                else:
                    ax2.plot(time[idx1:idx2+1], filtered_signal[idx1:idx2+1],
                            'lightblue', linewidth=2, alpha=0.5)
                    ax2.plot(time[idx2:idx3+1], filtered_signal[idx2:idx3+1],
                            'lightcoral', linewidth=2, alpha=0.5)
        
        # 再绘制保留的模式（深色）
        for pattern in patterns:
            i = pattern['index']
            if i+2 < len(turning_points):
                idx1 = turning_points[i-1]
                idx2 = turning_points[i]
                idx3 = turning_points[i+1]
                
                if pattern['fast_phase_first']:
                    ax2.plot(time[idx1:idx2+1], filtered_signal[idx1:idx2+1],
                            'red', linewidth=2, alpha=0.8)
                    ax2.plot(time[idx2:idx3+1], filtered_signal[idx2:idx3+1],
                            'blue', linewidth=2, alpha=0.8)
                else:
                    ax2.plot(time[idx1:idx2+1], filtered_signal[idx1:idx2+1],
                            'blue', linewidth=2, alpha=0.8)
                    ax2.plot(time[idx2:idx3+1], filtered_signal[idx2:idx3+1],
                            'red', linewidth=2, alpha=0.8)
        
        ax2.set_title(f'Detected Nystagmus Patterns (Direction: {direction}, SPV: {pattern_spv:.1f}°/s, CV: {cv:.1f}%)')
        ax2.set_ylabel('Position (°)')
        ax2.legend(['Signal', 'Filtered Fast Phase', 'Filtered Slow Phase', 'Fast Phase', 'Slow Phase'])
        ax2.grid(True)
        
        # 绘制斜率图
        ax3.scatter(slope_times, slopes, color='gray', s=20, alpha=0.3, label='All Velocities')
        if slow_times:
            ax3.scatter(slow_times, slow_velocities, color='red', s=30, label='Slow Phase')
        if fast_times:
            ax3.scatter(fast_times, fast_velocities, color='blue', s=30, label='Fast Phase')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        if spv is not None:
            title = f'Slope Analysis (Pattern: {pattern}, SPV: {spv:.1f}°/s)'
        else:
            title = 'Slope Analysis (No valid slow phase detected)'
        ax3.set_title(title)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (°/s)')
        ax3.legend()
        ax3.grid(True)
        ax3.set_ylim([-40, 40])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    return results 