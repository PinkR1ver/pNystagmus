import pandas as pd
import numpy as np

def calculate_cv(gains):
    """计算变异系数 (Coefficient of Variation)"""
    if isinstance(gains, str):
        gains = eval(gains)  # 只有在输入为字符串时才转换
    return np.std(gains) / np.mean(gains) * 100  # 转换为百分比

def filter_gains(gains_str, cv_threshold=5):
    """
    过滤掉导致变异系数超过阈值的数据点
    返回过滤后的gains列表和是否有数据被过滤的标志
    """
    gains = eval(gains_str)  # 原始数据是字符串，需要转换
    gains = np.array(gains)
    
    while True:
        cv = np.std(gains) / np.mean(gains) * 100
        if cv <= cv_threshold or len(gains) < 3:  # 保证至少保留3个数据点
            break
            
        # 计算每个点对变异系数的贡献
        temp_cvs = []
        for i in range(len(gains)):
            temp_gains = np.delete(gains, i)
            temp_cv = np.std(temp_gains) / np.mean(temp_gains) * 100
            temp_cvs.append(temp_cv)
        
        # 移除导致最大变异系数的点
        worst_point_idx = np.argmin(temp_cvs)
        gains = np.delete(gains, worst_point_idx)
    
    return gains.tolist()

# 读取原始数据
df = pd.read_csv('HIT/repeatability_analysis/HIT_data/hit_gain_analysis.csv')

# 处理左右眼的gain数据
for side in ['left', 'right']:
    # 计算原始变异系数
    df[f'{side}_original_cv'] = df[f'{side}_gains'].apply(calculate_cv)
    
    # 过滤gain数据
    df[f'{side}_filtered_gains'] = df[f'{side}_gains'].apply(filter_gains)
    
    # 计算新的统计数据
    df[f'{side}_filtered_mean'] = df[f'{side}_filtered_gains'].apply(np.mean)
    df[f'{side}_filtered_median'] = df[f'{side}_filtered_gains'].apply(np.median)
    df[f'{side}_filtered_std'] = df[f'{side}_filtered_gains'].apply(np.std)
    df[f'{side}_filtered_cv'] = df[f'{side}_filtered_gains'].apply(lambda x: np.std(x) / np.mean(x) * 100)
    df[f'{side}_filtered_count'] = df[f'{side}_filtered_gains'].apply(len)

# 保存结果
output_file = 'HIT/repeatability_analysis/HIT_data/hit_gain_analysis_filtered.csv'
df.to_csv(output_file, index=False)