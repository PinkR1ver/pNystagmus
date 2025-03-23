import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_gain_repeatability(csv_path):
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 1. 单次测试内的重复性分析 (Within-test Repeatability)
    def calculate_cv(gains):
        gains = eval(gains) if isinstance(gains, str) else gains
        return np.std(gains) / np.mean(gains) * 100
    
    df['left_cv'] = df['left_gains'].apply(calculate_cv)
    df['right_cv'] = df['right_gains'].apply(calculate_cv)
    
    # 2. 测试者内部一致性分析 (Intra-tester Consistency)
    # 只分析医生1的三次测试结果
    intra_tester = df[df['doctor_id'] == 1].pivot_table(
        index='patient_id',
        columns='test_number',
        values=['left_mean_gain', 'right_mean_gain'],
        aggfunc='first'
    ).round(3)
    
    # 添加均值和标准差
    intra_tester['left_mean'] = intra_tester['left_mean_gain'].mean(axis=1)
    intra_tester['left_std'] = intra_tester['left_mean_gain'].std(axis=1)
    intra_tester['right_mean'] = intra_tester['right_mean_gain'].mean(axis=1)
    intra_tester['right_std'] = intra_tester['right_mean_gain'].std(axis=1)
    
    # 3. 测试者间一致性分析 (Inter-tester Consistency)
    def calculate_inter_tester_icc(df):
        inter_tester_data = []
        for patient_id in df['patient_id'].unique():
            patient_data = df[df['patient_id'] == patient_id]
            tester1_data = patient_data[patient_data['doctor_id'] == 1]['left_mean_gain'].mean()
            tester2_data = patient_data[patient_data['doctor_id'] == 2]['left_mean_gain'].mean()
            if not (np.isnan(tester1_data) or np.isnan(tester2_data)):
                inter_tester_data.append([tester1_data, tester2_data])
        
        if not inter_tester_data:
            return np.nan
        
        measurements = np.array(inter_tester_data).T
        n = measurements.shape[1]
        between_subject_var = np.var(np.mean(measurements, axis=0))
        within_subject_var = np.mean(np.var(measurements, axis=1))
        
        icc = (between_subject_var - within_subject_var/n) / (between_subject_var + (n-1)*within_subject_var/n)
        return icc
    
    inter_tester = df.groupby(['patient_id', 'doctor_id'])[['left_mean_gain', 'right_mean_gain']].mean().round(3)
    inter_tester = inter_tester.unstack()
    inter_tester.columns = ['Tester1_Left', 'Tester2_Left', 'Tester1_Right', 'Tester2_Right']
    inter_tester['Left_Difference'] = inter_tester['Tester1_Left'] - inter_tester['Tester2_Left']
    inter_tester['Right_Difference'] = inter_tester['Tester1_Right'] - inter_tester['Tester2_Right']
    
    return {
        'within_test_cv': {
            'left': df['left_cv'].describe().round(2),
            'right': df['right_cv'].describe().round(2)
        },
        'intra_tester': intra_tester,
        'inter_tester': inter_tester,
        'inter_tester_icc': calculate_inter_tester_icc(df)
    }

def format_results_for_table(results):
    # 1. 单次测试内的变异系数表格
    cv_data = pd.DataFrame({
        'Eye': ['Left', 'Right'],
        'Mean CV(%)': [
            f"{results['within_test_cv']['left']['mean']:.2f}",
            f"{results['within_test_cv']['right']['mean']:.2f}"
        ],
        'Std CV(%)': [
            f"{results['within_test_cv']['left']['std']:.2f}",
            f"{results['within_test_cv']['right']['std']:.2f}"
        ],
        'Min CV(%)': [
            f"{results['within_test_cv']['left']['min']:.2f}",
            f"{results['within_test_cv']['right']['min']:.2f}"
        ],
        'Max CV(%)': [
            f"{results['within_test_cv']['left']['max']:.2f}",
            f"{results['within_test_cv']['right']['max']:.2f}"
        ]
    })
    
    # 2. 测试者内部一致性表格
    intra_tester = results['intra_tester'].copy()
    # 重命名列
    intra_tester.columns = [
        'Left_Test1', 'Left_Test2', 'Left_Test3',
        'Right_Test1', 'Right_Test2', 'Right_Test3',
        'Left_Mean', 'Left_Std',
        'Right_Mean', 'Right_Std'
    ]
    intra_tester_df = intra_tester.reset_index()
    intra_tester_df = intra_tester_df.round(2)  # 保留2位小数
    
    # 3. 测试者间一致性表格
    inter_tester = results['inter_tester'].copy()
    inter_tester_df = inter_tester.reset_index()
    # 重命名列并保留2位小数
    inter_tester_df.columns = ['Patient_ID', 
                              'Tester1_Left', 'Tester2_Left',
                              'Tester1_Right', 'Tester2_Right',
                              'Left_Difference', 'Right_Difference']
    inter_tester_df = inter_tester_df.round(2)  # 保留2位小数
    
    return {
        'cv_table': cv_data,
        'intra_tester_table': intra_tester_df,
        'inter_tester_table': inter_tester_df
    }

def create_three_line_table(df, title, output_path):
    """创建三线表并保存为图片"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 隐藏轴线
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 创建表格
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    loc='center',
                    cellLoc='center')
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 设置三线表样式
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
        cell.visible_edges = ''
        if row == 0:
            cell.visible_edges += 'TB'
        if row == len(df):
            cell.visible_edges += 'B'
        
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
    
    plt.title(title, pad=20, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置基础路径
    base_path = os.path.dirname(__file__)
    
    # 分析数据
    results = analyze_gain_repeatability(os.path.join(base_path, 'hit_gain_analysis.csv'))
    
    # 格式化结果
    formatted_results = format_results_for_table(results)
    
    # 生成表格图片并保存CSV
    # 1. Within-test Repeatability
    create_three_line_table(
        formatted_results['cv_table'],
        'Within-test Repeatability (CV Analysis)',
        os.path.join(base_path, 'within_test_repeatability.png')
    )
    formatted_results['cv_table'].to_csv(
        os.path.join(base_path, 'within_test_repeatability.csv'),
        index=False
    )
    
    # 2. Intra-tester Consistency
    create_three_line_table(
        formatted_results['intra_tester_table'],
        'Intra-tester Consistency Analysis',
        os.path.join(base_path, 'intra_tester_consistency.png')
    )
    formatted_results['intra_tester_table'].to_csv(
        os.path.join(base_path, 'intra_tester_consistency.csv'),
        index=False
    )
    
    # 3. Inter-tester Consistency
    create_three_line_table(
        formatted_results['inter_tester_table'],
        f'Inter-tester Consistency Analysis (ICC = {results["inter_tester_icc"]:.3f})',
        os.path.join(base_path, 'inter_tester_consistency.png')
    )
    formatted_results['inter_tester_table'].to_csv(
        os.path.join(base_path, 'inter_tester_consistency.csv'),
        index=False
    )

if __name__ == "__main__":
    main() 