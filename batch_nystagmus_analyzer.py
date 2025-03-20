import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import utils

def batch_analyze_nystagmus():
    """
    批量分析所有患者数据，生成眼震方向和SPV汇总表格
    """
    # 设置数据路径
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'phone')
    
    # 检查数据文件夹是否存在
    if not os.path.exists(data_path):
        print(f"错误：数据路径不存在 {data_path}")
        return
    
    # 获取所有患者文件夹
    patient_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    if not patient_folders:
        print("错误：未找到患者数据文件夹")
        return
    
    # 定义方向映射
    directions = ["left", "right", "up", "down"]
    direction_labels = {
        "left": "向左", 
        "right": "向右", 
        "up": "向上", 
        "down": "向下"
    }
    
    # 初始化结果列表
    results = []
    
    # 分析参数设置（使用默认值）
    highpass_parameter = {'cutoff': 0.1, 'fs': 60, 'order': 5}
    lowpass_parameter = {'cutoff': 6, 'fs': 60, 'order': 5}
    interpolate_ratio = 10
    
    # 循环分析每个患者的各个方向数据
    for patient_id in patient_folders:
        print(f"分析患者: {patient_id}")
        
        patient_result = {"患者编号": patient_id}
        
        for direction in directions:
            # 确定坐标轴方向
            direction_axis = "horizontal" if direction in ["left", "right"] else "vertical"
            
            # 构建可能的文件路径
            file_paths = [
                os.path.join(data_path, patient_id, direction, f"{patient_id}_{direction}_VOG.plist"),
                os.path.join(data_path, patient_id, direction, f"{patient_id}_{direction}.plist"),
                os.path.join(data_path, patient_id, f"{patient_id}_{direction}.plist")
            ]
            
            # 查找存在的文件路径
            file_path = None
            for path in file_paths:
                if os.path.exists(path):
                    file_path = path
                    break
            
            # 如果找不到文件，记录为NaN并继续
            if file_path is None:
                print(f"  - {direction_labels[direction]}: 未找到数据文件")
                patient_result[f"{direction_labels[direction]}_方向"] = "N/A"
                patient_result[f"{direction_labels[direction]}_SPV"] = np.nan
                continue
            
            try:
                # 预处理信号
                filtered_signal, time = utils.signal_preprocess(
                    file_path,
                    highpass_parameter=highpass_parameter,
                    lowpass_parameter=lowpass_parameter,
                    window_size=0,  # 跳过移动平均平滑
                    interpolate_ratio=interpolate_ratio
                )
                
                # 分析眼震模式
                patterns, filtered_patterns, nystagmus_direction, spv, cv = utils.identify_nystagmus_patterns(
                    filtered_signal, time, 
                    direction_axis=direction_axis
                )
                
                # 记录结果
                if patterns:
                    patient_result[f"{direction_labels[direction]}_方向"] = nystagmus_direction
                    patient_result[f"{direction_labels[direction]}_SPV"] = round(spv, 2)
                    print(f"  - {direction_labels[direction]}: 方向={nystagmus_direction}, SPV={spv:.2f}°/s, CV={cv:.2f}%")
                else:
                    patient_result[f"{direction_labels[direction]}_方向"] = "未检测到"
                    patient_result[f"{direction_labels[direction]}_SPV"] = np.nan
                    print(f"  - {direction_labels[direction]}: 未检测到有效眼震")
            
            except Exception as e:
                print(f"  - {direction_labels[direction]}: 分析错误: {str(e)}")
                patient_result[f"{direction_labels[direction]}_方向"] = "错误"
                patient_result[f"{direction_labels[direction]}_SPV"] = np.nan
        
        # 添加到结果列表
        results.append(patient_result)
    
    # 创建DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存为CSV文件
    output_file = "眼震分析结果汇总.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n分析完成！结果已保存至 {output_file}")
    
    return results_df

# 直接运行批处理
if __name__ == "__main__":
    batch_analyze_nystagmus() 