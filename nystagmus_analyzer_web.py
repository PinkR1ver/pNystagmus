import plistlib
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import utils  # 导入utils模块
import streamlit as st  # 导入streamlit库
import tempfile
import shutil
import zipfile

# 设置页面配置
st.set_page_config(layout="wide", page_title="眼震分析器")

# 页面标题
st.title("眼震分析器")

# 创建侧边栏参数控制
st.sidebar.header("参数设置")

# 设置临时目录来存储上传的文件
temp_dir = None

# 上传文件功能
st.header("上传数据")
st.write("请上传包含眼震数据的ZIP文件")
uploaded_file = st.file_uploader("选择ZIP文件", type="zip")

if uploaded_file is None:
    st.warning("请上传眼震数据ZIP文件进行分析")
    st.stop()

# 创建一个临时目录来解压文件
temp_dir = tempfile.mkdtemp()

try:
    # 保存上传的文件到临时位置
    temp_zip_path = os.path.join(temp_dir, "uploaded.zip")
    with open(temp_zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 解压文件
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # 列出解压后的文件夹
    uploaded_folders = [f for f in os.listdir(temp_dir) 
                      if os.path.isdir(os.path.join(temp_dir, f)) and f != "__MACOSX"]
    
    if not uploaded_folders:
        st.error("上传的ZIP文件中未找到文件夹，请确保ZIP文件包含正确的数据结构。")
        st.stop()
    
    st.success(f"成功上传并解压，找到 {len(uploaded_folders)} 个文件夹")
    selected_folder = st.selectbox("选择数据文件夹", uploaded_folders)
    data_path = temp_dir
    
except Exception as e:
    st.error(f"处理上传的文件时出错: {str(e)}")
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    st.stop()

# 修改为仅包含上下左右的视靶方向选择
target_direction = st.sidebar.radio("视靶方向", ["向左", "向右", "向上", "向下"])
# 根据视靶方向映射到文件夹名
direction_folder_map = {
    "向左": "left", 
    "向右": "right", 
    "向上": "up", 
    "向下": "down"
}
direction_folder = direction_folder_map[target_direction]

# 确定眼震方向判断使用的坐标轴
direction_axis = "horizontal" if target_direction in ["向左", "向右"] else "vertical"

# 构建文件路径
file_path = os.path.join(data_path, selected_folder, direction_folder, f"{selected_folder}_{direction_folder}_VOG.plist")

# 检查文件是否存在，如果不存在尝试几种可能的文件命名方式
if not os.path.exists(file_path):
    # 尝试其他可能的命名格式
    alternate_path_1 = os.path.join(data_path, selected_folder, direction_folder, f"{selected_folder}_{direction_folder}.plist")
    alternate_path_2 = os.path.join(data_path, selected_folder, f"{selected_folder}_{direction_folder}.plist")
    alternate_path_3 = os.path.join(data_path, selected_folder, direction_folder, "data.plist")
    
    if os.path.exists(alternate_path_1):
        file_path = alternate_path_1
    elif os.path.exists(alternate_path_2):
        file_path = alternate_path_2
    elif os.path.exists(alternate_path_3):
        file_path = alternate_path_3
    else:
        st.error(f"未找到与视靶方向 '{target_direction}' 匹配的数据文件。")
        st.error(f"已尝试路径: \n- {file_path}\n- {alternate_path_1}\n- {alternate_path_2}\n- {alternate_path_3}")
        st.stop()

# 预处理参数
st.sidebar.subheader("预处理参数")

# 高通滤波参数
st.sidebar.markdown("**高通滤波参数**")
highpass_cutoff = st.sidebar.slider("高通截止频率 (Hz)", 0.01, 1.0, 0.1, 0.01)
highpass_fs = st.sidebar.slider("高通采样频率 (Hz)", 30, 120, 60, 5)
highpass_order = st.sidebar.slider("高通滤波器阶数", 1, 10, 5, 1)

# 低通滤波参数
st.sidebar.markdown("**低通滤波参数**")
lowpass_cutoff = st.sidebar.slider("低通截止频率 (Hz)", 1.0, 20.0, 6.0, 0.5)
lowpass_fs = st.sidebar.slider("低通采样频率 (Hz)", 30, 120, 60, 5)
lowpass_order = st.sidebar.slider("低通滤波器阶数", 1, 10, 5, 1)

# 其他预处理参数
interpolate_ratio = st.sidebar.slider("插值比例", 1, 20, 10, 1)

# 拐点检测参数
st.sidebar.subheader("拐点检测参数")
prominence = st.sidebar.slider("峰值突出度", 0.01, 1.0, 0.1, 0.01)
distance = st.sidebar.slider("拐点最小距离", 50, 500, 150, 10)

# 眼震模式识别参数
st.sidebar.subheader("眼震模式识别参数")
min_time = st.sidebar.slider("最小时间阈值 (秒)", 0.1, 1.0, 0.3, 0.05)
max_time = st.sidebar.slider("最大时间阈值 (秒)", 0.5, 2.0, 0.8, 0.05)
min_ratio = st.sidebar.slider("最小斜率比例", 1.0, 3.0, 1.4, 0.1)
max_ratio = st.sidebar.slider("最大斜率比例", 3.0, 15.0, 8.0, 0.5)

# 设置参数字典
highpass_parameter = {'cutoff': highpass_cutoff, 'fs': highpass_fs, 'order': highpass_order}
lowpass_parameter = {'cutoff': lowpass_cutoff, 'fs': lowpass_fs, 'order': lowpass_order}

# 主程序部分
st.subheader("分析过程")

# 使用with语句创建一个进度条
with st.spinner("正在分析数据..."):
    # 加载数据
    timestamps, eye_angles = utils.plists_to_data(file_path)

    # 确保timestamps和eye_angles长度一致
    min_length = min(len(timestamps), len(eye_angles))
    timestamps = timestamps[:min_length]
    eye_angles = eye_angles[:min_length]

    # 创建一个大型Figure和多个子图
    fig = plt.figure(figsize=(18, 16))  # 减小高度

    # ---------- 1. 绘制前处理过程（合并原来的步骤，移除移动平均平滑） ----------
    plt.subplot(4, 1, 1)  # 使用4行1列的布局

    # 绘制原始数据
    plt.plot(timestamps, eye_angles, label='Original Data', alpha=0.7)

    # 应用高通滤波并绘制
    signal_highpass = utils.butter_highpass_filter(eye_angles, **highpass_parameter)
    plt.plot(timestamps, signal_highpass, label='After High-pass Filter', alpha=0.7)

    # 应用低通滤波并绘制 - 这是最终的预处理结果
    signal_lowpass = utils.butter_lowpass_filter(signal_highpass, **lowpass_parameter)
    plt.plot(timestamps, signal_lowpass, label='After Low-pass Filter', alpha=0.9, linewidth=2)

    plt.title('1. Signal Preprocessing Steps')
    plt.ylabel('Position (°)')
    plt.grid(True)
    plt.legend()

    # 重采样和完整预处理 - 修改参数传递，设置window_size=0以跳过移动平均平滑
    filtered_signal, time = utils.signal_preprocess(
        file_path,
        highpass_parameter=highpass_parameter,
        lowpass_parameter=lowpass_parameter,
        window_size=0,  # 设置为0以跳过移动平均平滑
        interpolate_ratio=interpolate_ratio
    )

    # ---------- 2. 拐点检测 ----------
    turning_points = utils.find_turning_points(filtered_signal, prominence=prominence, distance=distance)

    plt.subplot(4, 1, 2)  # 保持一致的subplot规格
    # 绘制原始信号（灰色，半透明背景）
    plt.plot(time, filtered_signal, 'gray', alpha=0.3, label='Original Signal')

    # 绘制拐点连接线
    plt.plot(time[turning_points], filtered_signal[turning_points], 'r-', label='Turning Points Connection', linewidth=2)

    # 突出显示拐点
    plt.plot(time[turning_points], filtered_signal[turning_points], 'ro', markersize=5, label='Turning Points')

    plt.title('2. Eye Movement Signal with Turning Points Connections')
    plt.ylabel('Position (°)')
    plt.legend()
    plt.grid(True)

    # ---------- 3. 斜率计算 ----------
    slope_times, slopes = utils.calculate_slopes(time, filtered_signal, turning_points)

    plt.subplot(4, 1, 3)  # 保持一致的subplot规格
    plt.scatter(slope_times, slopes, c='blue', s=30, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('3. Calculated Slopes between Turning Points')
    plt.ylabel('Slope (°/s)')
    plt.grid(True)
    plt.ylim([-40, 40])

    # ---------- 4. 眼震模式识别 ----------
    patterns, filtered_patterns, direction, pattern_spv, cv = utils.identify_nystagmus_patterns(
        filtered_signal, time, min_time=min_time, max_time=max_time, min_ratio=min_ratio, max_ratio=max_ratio,
        direction_axis=direction_axis
    )

    plt.subplot(4, 1, 4)  # 保持一致的subplot规格，修复错误
    plt.plot(time, filtered_signal, 'gray', alpha=0.5, label='Signal')

    # 先绘制被过滤掉的模式（浅色）
    for pattern in filtered_patterns:
        i = pattern['index']
        if i+2 < len(turning_points):
            idx1 = turning_points[i-1]
            idx2 = turning_points[i]
            idx3 = turning_points[i+1]
            
            # 注意：快相永远使用红色，慢相永远使用蓝色
            fast_segment = time[idx1:idx2+1] if pattern['fast_phase_first'] else time[idx2:idx3+1]
            slow_segment = time[idx2:idx3+1] if pattern['fast_phase_first'] else time[idx1:idx2+1]
            
            fast_signal = filtered_signal[idx1:idx2+1] if pattern['fast_phase_first'] else filtered_signal[idx2:idx3+1]
            slow_signal = filtered_signal[idx2:idx3+1] if pattern['fast_phase_first'] else filtered_signal[idx1:idx2+1]
            
            plt.plot(fast_segment, fast_signal, 'lightcoral', linewidth=2, alpha=0.5)
            plt.plot(slow_segment, slow_signal, 'lightblue', linewidth=2, alpha=0.5)

    # 再绘制保留的模式（深色）
    for pattern in patterns:
        i = pattern['index']
        if i+2 < len(turning_points):
            idx1 = turning_points[i-1]
            idx2 = turning_points[i]
            idx3 = turning_points[i+1]
            
            # 注意：快相永远使用红色，慢相永远使用蓝色
            fast_segment = time[idx1:idx2+1] if pattern['fast_phase_first'] else time[idx2:idx3+1]
            slow_segment = time[idx2:idx3+1] if pattern['fast_phase_first'] else time[idx1:idx2+1]
            
            fast_signal = filtered_signal[idx1:idx2+1] if pattern['fast_phase_first'] else filtered_signal[idx2:idx3+1]
            slow_signal = filtered_signal[idx2:idx3+1] if pattern['fast_phase_first'] else filtered_signal[idx1:idx2+1]
            
            plt.plot(fast_segment, fast_signal, 'red', linewidth=2, alpha=0.8)
            plt.plot(slow_segment, slow_signal, 'blue', linewidth=2, alpha=0.8)

    # 设置适当的方向标签
    direction_label = direction
    if direction_axis == "vertical":
        if direction == "left":
            direction_label = "up"
        elif direction == "right":
            direction_label = "down"

    plt.title(f'4. Pattern-Based Nystagmus Analysis (Direction: {direction_label}, SPV: {pattern_spv:.1f}°/s, CV: {cv:.1f}%)')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (°)')
    plt.legend(['Signal', 'Filtered Fast Phase', 'Filtered Slow Phase', 'Fast Phase (Red)', 'Slow Phase (Blue)'])
    plt.grid(True)

    # 调整整体布局 - 使用subplots_adjust而不是tight_layout
    plt.subplots_adjust(hspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.05)

    # 使用Streamlit展示图形
    st.pyplot(fig)

    # 输出分析结果
    st.subheader("眼震分析结果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("检测到的眼震模式数量", f"{len(patterns)}")
        # 使用修正后的方向标签
        st.metric("眼震方向", f"{direction_label}")
    
    with col2:
        st.metric("慢相速度(SPV)", f"{pattern_spv:.2f}°/s")
        st.metric("变异系数(CV)", f"{cv:.2f}%")
    
    if len(patterns) > 0:
        avg_ratio = np.mean([p['ratio'] for p in patterns])
        fast_first_count = sum(1 for p in patterns if p['fast_phase_first'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("平均快慢相比例", f"{avg_ratio:.2f}")
        with col2:
            st.metric("快相在前的模式数", f"{fast_first_count} ({fast_first_count/len(patterns)*100:.1f}%)")

    # 生成并提供结果下载
    st.subheader("下载分析结果")
    
    # 准备结果字典
    results = {
        "分析时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "数据文件夹": selected_folder,
        "视靶方向": target_direction,
        "检测到的眼震模式数量": len(patterns),
        "眼震方向": direction_label,
        "慢相速度(SPV)": f"{pattern_spv:.2f}°/s",
        "变异系数(CV)": f"{cv:.2f}%"
    }
    
    if len(patterns) > 0:
        results["平均快慢相比例"] = f"{avg_ratio:.2f}"
        results["快相在前的模式数"] = f"{fast_first_count} ({fast_first_count/len(patterns)*100:.1f}%)"
    
    # 将结果转换为CSV格式
    import pandas as pd
    results_df = pd.DataFrame(list(results.items()), columns=['参数', '值'])
    csv = results_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="下载分析结果CSV",
        data=csv,
        file_name=f"眼震分析_{selected_folder}_{target_direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
    
    # 保存分析图表
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    st.download_button(
        label="下载分析图表",
        data=buf.getvalue(),
        file_name=f"眼震分析图_{selected_folder}_{target_direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png",
    )

# 清理临时文件
if temp_dir and os.path.exists(temp_dir):
    try:
        shutil.rmtree(temp_dir)
    except:
        pass  # 忽略清理错误

# 添加应用程序说明
st.sidebar.markdown("---")
st.sidebar.subheader("关于应用")
st.sidebar.info(
    "此眼震分析器允许您通过调整侧边栏中的参数来分析眼震数据。\n\n"
    "请上传包含眼震数据的ZIP文件。\n\n"
    "视靶方向指的是诱发眼震的视觉刺激的移动方向。"
    "对于上下视靶，眼震方向将判断为上/下；对于左右视靶，眼震方向将判断为左/右。\n\n"
    "调整参数后，结果将自动更新。"
)

st.sidebar.markdown("---")
st.sidebar.subheader("数据格式说明")
st.sidebar.info(
    "上传的ZIP文件应包含以下结构：\n"
    "```\n"
    "└── 患者文件夹/\n"
    "    ├── left/\n"
    "    │   └── 数据文件.plist\n"
    "    ├── right/\n"
    "    │   └── 数据文件.plist\n"
    "    ├── up/\n"
    "    │   └── 数据文件.plist\n"
    "    └── down/\n"
    "        └── 数据文件.plist\n"
    "```\n"
    "数据文件名可以是：\n"
    "- [文件夹名]_[方向]_VOG.plist\n"
    "- [文件夹名]_[方向].plist\n"
    "- data.plist"
)

st.sidebar.markdown("---")
st.sidebar.subheader("示例ZIP文件")
st.sidebar.markdown("""
如果您需要示例数据，请下载[示例ZIP文件](https://example.com/sample_nystagmus_data.zip)进行测试。
""")