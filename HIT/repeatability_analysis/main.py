import os
import numpy as np
import pandas as pd
from pathlib import Path
import plistlib
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

class HITGainAnalysis:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def parse_filename(self, filename):
        """解析文件名信息"""
        base_name = filename.split('_')[0]
        parts = base_name.split('-')

        info = {
            'date': parts[0][:8],
            'patient_id': parts[0][8:],
            'doctor_id': parts[1],
            'test_number': parts[2]
        }
        return info

    def load_plist_data(self, file_path):
        """加载plist文件数据"""
        with open(file_path, 'rb') as f:
            data = plistlib.load(f)
        return np.array(data)

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        """创建带通滤波器"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def moving_average(self, data, window_size):
        """应用移动平均"""
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='same')

    def signal_preprocess(self, file_path, highpass_parameter, lowpass_parameter, window_size):
        """信号预处理，参考原main.py的处理方法"""
        # 加载数据
        raw_data = self.load_plist_data(file_path)

        # 应用带通滤波
        fs = 60  # 采样率
        b, a = self.butter_bandpass(
            highpass_parameter['cutoff'],
            lowpass_parameter['cutoff'],
            fs,
            order=5
        )
        filtered_data = filtfilt(b, a, raw_data)

        # 应用移动平均
        smoothed_data = self.moving_average(filtered_data, window_size)

        # 生成时间序列
        time = np.arange(len(smoothed_data)) / fs

        return smoothed_data, time

    def velocity_calculation(self, position_data, fs=60):
        """计算速度，使用差分"""
        velocity = np.diff(position_data) * fs
        velocity = np.append(velocity, velocity[-1])
        return velocity, np.arange(len(velocity)) / fs

    def velocity_peak_detection(self, velocity_data, threshold=50):
        """检测速度峰值"""
        # 检测左向峰值（负峰值）
        left_peaks, _ = find_peaks(-velocity_data, height=threshold)
        # 检测右向峰值（正峰值）
        right_peaks, _ = find_peaks(velocity_data, height=threshold)

        return left_peaks, right_peaks

    def calculate_gain(self, head_velocity, eye_velocity, peak_idx, window_size=10):
        """计算单个峰值处的增益值"""
        start_idx = max(0, peak_idx - window_size)
        end_idx = min(len(head_velocity), peak_idx + window_size)

        head_peak = np.max(np.abs(head_velocity[start_idx:end_idx]))
        eye_peak = np.max(np.abs(eye_velocity[start_idx:end_idx]))

        return eye_peak / head_peak if head_peak != 0 else np.nan

    def analyze_test(self, test_file):
        """分析单次测试数据"""
        # 解析文件信息
        test_file = test_file.split('/')[-1]
        info = self.parse_filename(test_file)

        # 构建文件路径
        base_path = self.data_dir / info['patient_id']
        head_path = base_path / test_file
        eye_path = base_path / test_file.replace('head', 'lefteye')

        # 滤波器参数
        highpass_params = {
            'cutoff': 0.1,
            'fs': 60,
            'order': 5
        }

        lowpass_params = {
            'cutoff': 6.0,
            'fs': 60,
            'order': 5
        }

        window_size = 5

        # 预处理信号
        head_data, time_head = self.signal_preprocess(
            head_path,
            highpass_params,
            lowpass_params,
            window_size
        )

        eye_data, time_eye = self.signal_preprocess(
            eye_path,
            highpass_params,
            lowpass_params,
            window_size
        )

        # 计算速度
        head_velocity, time_head_vel = self.velocity_calculation(head_data)
        eye_velocity, time_eye_vel = self.velocity_calculation(eye_data)

        # 检测峰值
        left_peaks, right_peaks = self.velocity_peak_detection(head_velocity)

        # 计算增益值
        left_gains = [self.calculate_gain(head_velocity, eye_velocity, peak)
                     for peak in left_peaks]
        right_gains = [self.calculate_gain(head_velocity, eye_velocity, peak)
                      for peak in right_peaks]

        # 过滤无效的增益值
        left_gains = [g for g in left_gains if not np.isnan(g)]
        right_gains = [g for g in right_gains if not np.isnan(g)]

        # 创建结果数据框
        results = {
            'date': info['date'],
            'patient_id': info['patient_id'],
            'doctor_id': info['doctor_id'],
            'test_number': info['test_number'],
            'left_gains': [float(g) for g in left_gains],
            'right_gains': [float(g) for g in right_gains],
            'left_mean_gain': float(np.mean(left_gains)) if left_gains else np.nan,
            'right_mean_gain': float(np.mean(right_gains)) if right_gains else np.nan,
            'left_std_gain': float(np.std(left_gains)) if left_gains else np.nan,
            'right_std_gain': float(np.std(right_gains)) if right_gains else np.nan,
            'left_count': len(left_gains),
            'right_count': len(right_gains)
        }

        return pd.DataFrame([results])

def main():
    # 设置数据目录
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, "HIT_data")

    # 初始化分析器
    analyzer = HITGainAnalysis(data_dir)

    # 获取所有测试文件
    all_results = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith("_head.plist"):
                print(f"Processing {file}...")
                results = analyzer.analyze_test(os.path.join(root, file))
                all_results.append(results)

    # 合并所有结果
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)

        # 保存结果到base_path
        output_path = os.path.join(base_path, 'hit_gain_analysis.csv')
        final_results.to_csv(output_path, index=False)

        # 打印汇总统计
        print("\nAnalysis Summary:")
        print(f"Total tests analyzed: {len(final_results)}")
        print("\nMean Gains by Patient:")
        summary = final_results.groupby('patient_id').agg({
            'left_mean_gain': ['mean', 'std'],
            'right_mean_gain': ['mean', 'std']
        })
        print(summary)
    else:
        print("No data files found to analyze.")

if __name__ == "__main__":
    main()