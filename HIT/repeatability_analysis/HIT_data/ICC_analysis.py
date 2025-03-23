import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg

def calculate_icc(df):
    """计算ICC（组内相关系数）"""
    # 只处理医生1的数据
    doctor_data = df[df['doctor_id'] == 1]
    
    # 获取该医生测试的所有病人和测试次数
    patients = doctor_data['patient_id'].unique()
    test_numbers = doctor_data['test_number'].unique()
    
    icc_results = {}
    if len(patients) >= 5:  # 只处理患者数量大于等于5的数据
        # 创建评分数据
        data_for_icc = []
        data_for_icc_sorted = []  # 新增：用于存储排序后的数据
        target_counter = 0
        
        for patient in patients:
            patient_data = doctor_data[doctor_data['patient_id'] == patient]
            
            # 处理左侧甩头数据
            left_gains = []
            for test in test_numbers:
                test_data = patient_data[patient_data['test_number'] == test]
                if len(test_data) > 0:
                    left_gain = test_data['left_filtered_median'].values[0]
                    if not np.isnan(left_gain):
                        left_gains.append(left_gain)
            
            # 处理右侧甩头数据
            right_gains = []
            for test in test_numbers:
                test_data = patient_data[patient_data['test_number'] == test]
                if len(test_data) > 0:
                    right_gain = test_data['right_filtered_median'].values[0]
                    if not np.isnan(right_gain):
                        right_gains.append(right_gain)
            
            # 如果左右两侧都有完整的3次测试数据
            if len(left_gains) == 3 and len(right_gains) == 3:
                # 原始数据
                for rater, rating in enumerate(left_gains):
                    data_for_icc.append({
                        'targets': target_counter,
                        'raters': rater,
                        'ratings': rating
                    })
                
                # 排序后的数据
                sorted_left_gains = sorted(left_gains, reverse=True)  # 从大到小排序
                for rater, rating in enumerate(sorted_left_gains):
                    data_for_icc_sorted.append({
                        'targets': target_counter,
                        'raters': rater,
                        'ratings': rating
                    })
                target_counter += 1
                
                # 右侧甩头数据处理（同样进行排序）
                for rater, rating in enumerate(right_gains):
                    data_for_icc.append({
                        'targets': target_counter,
                        'raters': rater,
                        'ratings': rating
                    })
                
                sorted_right_gains = sorted(right_gains, reverse=True)  # 从大到小排序
                for rater, rating in enumerate(sorted_right_gains):
                    data_for_icc_sorted.append({
                        'targets': target_counter,
                        'raters': rater,
                        'ratings': rating
                    })
                target_counter += 1
        
        if len(data_for_icc) >= 30:  # 确保有足够的数据
            try:
                # 原始数据的ICC
                data_for_icc = pd.DataFrame(data_for_icc)
                icc = pg.intraclass_corr(data=data_for_icc, 
                                       targets='targets', 
                                       raters='raters', 
                                       ratings='ratings')
                
                # 排序后数据的ICC
                data_for_icc_sorted = pd.DataFrame(data_for_icc_sorted)
                icc_sorted = pg.intraclass_corr(data=data_for_icc_sorted, 
                                              targets='targets', 
                                              raters='raters', 
                                              ratings='ratings')
                
                # 保存ICC结果
                icc_results['医生1'] = {
                    'ICC(原始)': icc.loc[icc['Type'] == 'ICC3', 'ICC'].values[0],
                    '95%CI下限(原始)': icc.loc[icc['Type'] == 'ICC3', 'CI95%'].values[0][0],
                    '95%CI上限(原始)': icc.loc[icc['Type'] == 'ICC3', 'CI95%'].values[0][1],
                    'ICC(排序后)': icc_sorted.loc[icc_sorted['Type'] == 'ICC3', 'ICC'].values[0],
                    '95%CI下限(排序后)': icc_sorted.loc[icc_sorted['Type'] == 'ICC3', 'CI95%'].values[0][0],
                    '95%CI上限(排序后)': icc_sorted.loc[icc_sorted['Type'] == 'ICC3', 'CI95%'].values[0][1],
                    '测试病人数': len(patients),
                    '有效病人数': target_counter // 2,
                    '测试次数': 3
                }
                
                # 打印数据框以供检查
                print("\n原始数据:")
                print(data_for_icc)
                print("\n排序后数据:")
                print(data_for_icc_sorted)
                
            except Exception as e:
                print(f"处理医生1的数据时出错: {str(e)}")
                icc_results['医生1'] = {
                    'ICC(原始)': np.nan,
                    'ICC(排序后)': np.nan,
                    '测试病人数': len(patients),
                    '有效病人数': target_counter // 2,
                    '备注': f'计算出错: {str(e)}'
                }
        else:
            icc_results['医生1'] = {
                'ICC(原始)': np.nan,
                'ICC(排序后)': np.nan,
                '测试病人数': len(patients),
                '有效病人数': target_counter // 2,
                '备注': '有效数据不足'
            }
    else:
        icc_results['医生1'] = {
            'ICC(原始)': np.nan,
            'ICC(排序后)': np.nan,
            '测试病人数': len(patients),
            '有效病人数': 0,
            '备注': '患者数量不足（少于5人）'
        }
    
    return icc_results

# 读取数据
df = pd.read_csv('HIT/repeatability_analysis/HIT_data/hit_gain_analysis_filtered.csv')

# 计算ICC
icc_results = calculate_icc(df)

# 打印结果
print("医生1的测试结果：")
for doctor, results in icc_results.items():
    print(f"\n{doctor}:")
    for key, value in results.items():
        if isinstance(value, float):
            if np.isnan(value):
                print(f"{key}: NA")
            else:
                print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

# 保存结果到CSV
results_df = pd.DataFrame({
    '医生ID': [],
    'ICC(原始)': [],
    '95%CI下限(原始)': [],
    '95%CI上限(原始)': [],
    'ICC(排序后)': [],
    '95%CI下限(排序后)': [],
    '95%CI上限(排序后)': [],
    '测试病人数': [],
    '有效病人数': [],
    '备注': []
})

for doctor, results in icc_results.items():
    results_df = pd.concat([results_df, pd.DataFrame({
        '医生ID': [doctor],
        'ICC(原始)': [results.get('ICC(原始)', np.nan)],
        '95%CI下限(原始)': [results.get('95%CI下限(原始)', np.nan)],
        '95%CI上限(原始)': [results.get('95%CI上限(原始)', np.nan)],
        'ICC(排序后)': [results.get('ICC(排序后)', np.nan)],
        '95%CI下限(排序后)': [results.get('95%CI下限(排序后)', np.nan)],
        '95%CI上限(排序后)': [results.get('95%CI上限(排序后)', np.nan)],
        '测试病人数': [results.get('测试病人数', 0)],
        '有效病人数': [results.get('有效病人数', 0)],
        '备注': [results.get('备注', '')]
    })])

results_df.to_csv('HIT/repeatability_analysis/HIT_data/icc_analysis_results.csv', index=False)
