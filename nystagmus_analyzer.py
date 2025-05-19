import plistlib
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import utils  # 导入utils模块
import streamlit as st  # 导入streamlit库
from new_plist_parser import parse_new_plist # 导入新的plist解析函数

# 设置页面配置
st.set_page_config(layout="wide", page_title="Nystagmus Analyzer")

# 页面标题
st.title("Nystagmus Analyzer")

# 创建侧边栏参数控制
st.sidebar.header("Parameter Settings")

# 数据加载设置
st.sidebar.subheader("Data Loading")
uploaded_file = st.sidebar.file_uploader("Upload .plist data file", type=["plist"])

# 选择分析轴
analysis_axis_option = st.sidebar.radio("Select Analysis Axis", ("Horizontal (X-axis)", "Vertical (Y-axis)"))

# 根据选择设置 direction_axis 和 plist data key
if analysis_axis_option == "Horizontal (X-axis)":
    direction_axis = "horizontal"
    axis_data_key = "LeftEyeXDegList"
elif analysis_axis_option == "Vertical (Y-axis)":
    direction_axis = "vertical"
    axis_data_key = "LeftEyeYDegList"

# 预处理参数
st.sidebar.subheader("Preprocessing Parameters")

# 高通滤波参数
st.sidebar.markdown("**High-pass Filter Parameters**")
highpass_cutoff = st.sidebar.slider("High-pass Cutoff (Hz)", 0.01, 1.0, 0.1, 0.01)
highpass_fs = st.sidebar.slider("High-pass Sampling Freq (Hz)", 30, 120, 60, 5)
highpass_order = st.sidebar.slider("High-pass Filter Order", 1, 10, 5, 1)

# 低通滤波参数
st.sidebar.markdown("**Low-pass Filter Parameters**")
lowpass_cutoff = st.sidebar.slider("Low-pass Cutoff (Hz)", 1.0, 20.0, 6.0, 0.5)
lowpass_fs = st.sidebar.slider("Low-pass Sampling Freq (Hz)", 30, 120, 60, 5)
lowpass_order = st.sidebar.slider("Low-pass Filter Order", 1, 10, 5, 1)

# 其他预处理参数
interpolate_ratio = st.sidebar.slider("Interpolation Ratio", 1, 20, 10, 1)

# 拐点检测参数
st.sidebar.subheader("Turning Point Detection Parameters")
prominence = st.sidebar.slider("Peak Prominence", 0.01, 1.0, 0.1, 0.01)
distance = st.sidebar.slider("Min Distance Between Points", 50, 500, 150, 10)

# 眼震模式识别参数
st.sidebar.subheader("Nystagmus Pattern Recognition Parameters")
min_time = st.sidebar.slider("Min Time Threshold (s)", 0.1, 1.0, 0.3, 0.05)
max_time = st.sidebar.slider("Max Time Threshold (s)", 0.5, 2.0, 0.8, 0.05)
min_ratio = st.sidebar.slider("Min Slope Ratio", 1.0, 3.0, 1.4, 0.1)
max_ratio = st.sidebar.slider("Max Slope Ratio", 3.0, 15.0, 8.0, 0.5)

# 设置参数字典
highpass_parameter = {'cutoff': highpass_cutoff, 'fs': highpass_fs, 'order': highpass_order}
lowpass_parameter = {'cutoff': lowpass_cutoff, 'fs': lowpass_fs, 'order': lowpass_order}

if uploaded_file is not None:
    st.subheader("Analysis Process")
    # 使用with语句创建一个进度条
    with st.spinner("Analyzing data..."):
        try:
            timestamps, eye_angles, available_keys = parse_new_plist(uploaded_file, axis_data_key=axis_data_key)
        except ValueError as e: # Catch specific timestamp parsing errors
            st.error(f"File parsing error (timestamp format issue): {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unknown error during file parsing: {e}")
            st.stop()

        if eye_angles.size == 0:
            st.error(f"Could not load data for '{axis_data_key}' from the file, or data is empty.")
            if available_keys:
                st.info(f"Available eye movement data list keys in file: {available_keys}. Please check if the selected axis is correct or check the file content.")
            else:
                st.info("No eye movement data keys ending with 'DegList' found in the file.")
            st.stop()
        
        if timestamps.size == 0:
            st.error("Could not load timestamp data (TimeList) from the file, or data is empty.")
            st.stop()

        min_length = min(len(timestamps), len(eye_angles))
        timestamps = timestamps[:min_length]
        eye_angles = eye_angles[:min_length]

        # 创建一个大型Figure和多个子图
        fig = plt.figure(figsize=(18, 16))

        # ---------- 1. 绘制前处理过程（合并原来的步骤，移除移动平均平滑） ----------
        plt.subplot(4, 1, 1)

        # 绘制原始数据
        plt.plot(timestamps, eye_angles, label=f'Original Data ({axis_data_key})', alpha=0.7)

        # 应用高通滤波并绘制
        signal_highpass = utils.butter_highpass_filter(eye_angles, **highpass_parameter)
        plt.plot(timestamps, signal_highpass, label='After High-pass Filter', alpha=0.7)

        # 应用低通滤波并绘制 - 这是最终的预处理结果
        signal_lowpass = utils.butter_lowpass_filter(signal_highpass, **lowpass_parameter)
        plt.plot(timestamps, signal_lowpass, label='After Low-pass Filter', alpha=0.9, linewidth=2)

        plt.title(f'1. Signal Preprocessing Steps ({analysis_axis_option})')
        plt.ylabel('Position (°)')
        plt.grid(True)
        plt.legend()

        # 重采样和完整预处理 - 使用解析得到的数据
        filtered_signal, time = utils.signal_preprocess(
            timestamps,
            eye_angles,
            highpass_parameter=highpass_parameter,
            lowpass_parameter=lowpass_parameter,
            window_size=0,
            interpolate_ratio=interpolate_ratio
        )

        if filtered_signal.size == 0 or time.size == 0:
            st.error("预处理后信号为空，请检查参数或输入数据。")
            st.stop()

        # ---------- 2. 拐点检测 ----------
        turning_points = utils.find_turning_points(filtered_signal, prominence=prominence, distance=distance)

        plt.subplot(4, 1, 2)
        plt.plot(time, filtered_signal, 'gray', alpha=0.3, label='Original Signal')
        plt.plot(time[turning_points], filtered_signal[turning_points], 'r-', label='Turning Points Connection', linewidth=2)
        plt.plot(time[turning_points], filtered_signal[turning_points], 'ro', markersize=5, label='Turning Points')
        plt.title(f'2. Eye Movement Signal with Turning Points ({analysis_axis_option})')
        plt.ylabel('Position (°)')
        plt.legend()
        plt.grid(True)

        # ---------- 3. 斜率计算 ----------
        slope_times, slopes = utils.calculate_slopes(time, filtered_signal, turning_points)

        plt.subplot(4, 1, 3)
        plt.scatter(slope_times, slopes, c='blue', s=30, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title(f'3. Calculated Slopes ({analysis_axis_option})')
        plt.ylabel('Slope (°/s)')
        plt.grid(True)
        plt.ylim([-40, 40])

        # ---------- 4. 眼震模式识别 ----------
        patterns, filtered_patterns, direction, pattern_spv, cv = utils.identify_nystagmus_patterns(
            filtered_signal, time, min_time=min_time, max_time=max_time, min_ratio=min_ratio, max_ratio=max_ratio,
            direction_axis=direction_axis
        )

        plt.subplot(4, 1, 4)
        plt.plot(time, filtered_signal, 'gray', alpha=0.5, label='Signal')

        for pattern_item in filtered_patterns:
            i = pattern_item['index']
            if i > 0 and i + 1 < len(turning_points):
                idx1 = turning_points[i-1]
                idx2 = turning_points[i]
                idx3 = turning_points[i+1]
                fast_segment = time[idx1:idx2+1] if pattern_item['fast_phase_first'] else time[idx2:idx3+1]
                slow_segment = time[idx2:idx3+1] if pattern_item['fast_phase_first'] else time[idx1:idx2+1]
                fast_signal = filtered_signal[idx1:idx2+1] if pattern_item['fast_phase_first'] else filtered_signal[idx2:idx3+1]
                slow_signal = filtered_signal[idx2:idx3+1] if pattern_item['fast_phase_first'] else filtered_signal[idx1:idx2+1]
                plt.plot(fast_segment, fast_signal, 'lightcoral', linewidth=2, alpha=0.5)
                plt.plot(slow_segment, slow_signal, 'lightblue', linewidth=2, alpha=0.5)

        for pattern_item in patterns:
            i = pattern_item['index']
            if i > 0 and i + 1 < len(turning_points):
                idx1 = turning_points[i-1]
                idx2 = turning_points[i]
                idx3 = turning_points[i+1]
                fast_segment = time[idx1:idx2+1] if pattern_item['fast_phase_first'] else time[idx2:idx3+1]
                slow_segment = time[idx2:idx3+1] if pattern_item['fast_phase_first'] else time[idx1:idx2+1]
                fast_signal = filtered_signal[idx1:idx2+1] if pattern_item['fast_phase_first'] else filtered_signal[idx2:idx3+1]
                slow_signal = filtered_signal[idx2:idx3+1] if pattern_item['fast_phase_first'] else filtered_signal[idx1:idx2+1]
                plt.plot(fast_segment, fast_signal, 'red', linewidth=2, alpha=0.8)
                plt.plot(slow_segment, slow_signal, 'blue', linewidth=2, alpha=0.8)

        # Determine direction label in English for plot title
        english_direction_label = str(direction).capitalize() # Default
        if direction_axis == "horizontal":
            if direction == "left": english_direction_label = "Left"
            elif direction == "right": english_direction_label = "Right"
        elif direction_axis == "vertical":
            # For vertical, 'left' from identify_nystagmus_patterns typically means 'Up', and 'right' means 'Down'
            if direction == "left": english_direction_label = "Up" 
            elif direction == "right": english_direction_label = "Down" 

        plt.title(f'4. Pattern-Based Nystagmus Analysis ({analysis_axis_option} - Direction: {english_direction_label}, SPV: {pattern_spv:.1f}°/s, CV: {cv:.1f}%)')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (°)')
        legend_items = ['Signal']
        if any(filtered_patterns): legend_items.extend(['Filtered Fast Phase', 'Filtered Slow Phase'])
        if any(patterns): legend_items.extend(['Fast Phase (Red)', 'Slow Phase (Blue)'])
        plt.legend(legend_items)
        plt.grid(True)

        plt.subplots_adjust(hspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.05)
        st.pyplot(fig)

        st.subheader(f"Nystagmus Analysis Results ({analysis_axis_option})")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Nystagmus Patterns", f"{len(patterns)}")
            st.metric("Nystagmus Direction", f"{english_direction_label}")
        with col2:
            st.metric("Slow Phase Velocity (SPV)", f"{pattern_spv:.2f}°/s")
            st.metric("Coefficient of Variation (CV)", f"{cv:.2f}%")
        
        if len(patterns) > 0:
            avg_ratio_values = [p['ratio'] for p in patterns if 'ratio' in p and p['ratio'] is not None]
            avg_ratio = np.mean(avg_ratio_values) if avg_ratio_values else 0.0
            fast_first_count = sum(1 for p_item in patterns if p_item.get('fast_phase_first', False))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Fast/Slow Phase Ratio", f"{avg_ratio:.2f}")
            with col2:
                st.metric("Patterns with Fast Phase First", f"{fast_first_count} ({fast_first_count/len(patterns)*100:.1f}%)")

else:
    st.info("Please upload a .plist file and select the analysis axis in the sidebar to begin analysis.")

# 添加应用程序说明
st.sidebar.markdown("---")
st.sidebar.subheader("About this App")
st.sidebar.info(
    "This Nystagmus Analyzer allows you to upload a .plist file and analyze nystagmus data by adjusting parameters."
    "You can select horizontal or vertical eye movement data for analysis."
    "Results will update automatically."
)


