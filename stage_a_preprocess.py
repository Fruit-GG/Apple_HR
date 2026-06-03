#!/usr/bin/env python3
"""
Stage A: Data Preprocessing
================================
把 test_measure.csv 从长表转成 workout 级别的宽表。

输入：
  - data/subject-info.csv：个体静态信息
  - data/test_measure.csv：时间序列数据（长表）

输出：
  - data/processed_data.feather：处理后的宽表（每行一个 workout）
  - data/data_quality_report.txt：数据质量检查报告
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 数据读取
# ============================================================================
print("=" * 80)
print("STAGE A: 数据整理 - 读取原始数据")
print("=" * 80)

subject_info = pd.read_csv('data/subject-info.csv')
test_measure = pd.read_csv('data/test_measure.csv')

print(f"\n✓ subject-info.csv: {subject_info.shape[0]} 行")
print(f"✓ test_measure.csv: {test_measure.shape[0]} 行")
print(f"\n 列名 (subject-info):\n  {list(subject_info.columns)}")
print(f"\n 列名 (test_measure):\n  {list(test_measure.columns)}")

# ============================================================================
# 2. 初步检查
# ============================================================================
print("\n" + "=" * 80)
print("初步数据检查")
print("=" * 80)

n_subjects = test_measure['ID'].nunique()
n_workouts = test_measure['ID_test'].nunique()

print(f"\n✓ 独特 subject (ID): {n_subjects}")
print(f"✓ 独特 workouts (ID_test): {n_workouts}")

# 检查缺失值
print(f"\n缺失值统计 (test_measure):")
missing_counts = test_measure.isnull().sum()
if missing_counts.sum() == 0:
    print("  ✓ 无缺失值")
else:
    print(missing_counts[missing_counts > 0])

# ============================================================================
# 3. 按 ID_test 分组，构造 workout 级别数据
# ============================================================================
print("\n" + "=" * 80)
print("数据分组与处理")
print("=" * 80)

# 先合并 subject-info 的信息到 test_measure
# 注意：subject_info 中每个 ID_test 只对应一个 ID，所以要小心不要创建笛卡尔积
# 先按 ID_test 找到对应的 subject 特征
subject_feature_lookup = {}
for _, row in subject_info.iterrows():
    subject_feature_lookup[row['ID_test']] = {
        'ID': row['ID'],
        'Age': row['Age'],
        'Sex': row['Sex'],
        'Weight': row['Weight'],
        'Height': row['Height'],
        'Humidity': row['Humidity'],
        'Temperature': row['Temperature']
    }

test_measure_with_info = test_measure.copy()
for col in ['Age', 'Sex', 'Weight', 'Height', 'Humidity', 'Temperature']:
    test_measure_with_info[col] = test_measure_with_info['ID_test'].map(
        lambda x: subject_feature_lookup.get(x, {}).get(col, np.nan)
    )

# 按 ID_test 分组
grouped = test_measure_with_info.groupby('ID_test')

print(f"\n分组完成，共 {grouped.ngroups} 个 workouts")

# ============================================================================
# 4. 对每个 workout，进行插值和构造宽表
# ============================================================================
print("\nStatus: 处理每个 workout...")

def process_workout(group):
    """
    对单个 workout：
    1. 按时间排序
    2. 检查异常
    3. 插值到统一网格
    4. 返回一行数据（序列存为列表）
    """
    
    # 排序
    group = group.sort_values('time').reset_index(drop=True)
    
    # 检查最小长度
    if len(group) < 5:
        return None
    
    time_vals = group['time'].values.astype(float)
    
    # 检查时间单调性
    if not np.all(np.diff(time_vals) >= 0):
        # 有重复或倒序的时间
        return None
    
    # 移除时间完全重复的点（只保留第一个）
    unique_times_idx = np.concatenate([[True], np.diff(time_vals) > 0])
    group = group[unique_times_idx].reset_index(drop=True)
    
    if len(group) < 5:
        return None
    
    time_vals = group['time'].values.astype(float)
    
    # ========== 插值 =========
    # 统一时间网格：从开始到结束，10秒间隔（仿照原论文）
    t_start = time_vals[0]
    t_end = time_vals[-1]
    duration = t_end - t_start
    
    # 如果 workout 太短，跳过
    if duration < 30:
        return None
    
    # 创建统一时间网格（10秒间隔）
    target_grid = np.arange(t_start, t_end + 1, 10.0)
    
    # 如果网格太短，跳过
    if len(target_grid) < 5:
        return None
    
    # 为各列进行线性插值
    result = {}
    result['ID_test'] = group['ID_test'].iloc[0]
    result['ID'] = group['ID'].iloc[0]
    result['duration'] = duration
    result['n_measurements'] = len(group)
    result['n_interpolated_points'] = len(target_grid)
    
    # 取第一行的主体特征
    result['Age'] = group['Age'].iloc[0]
    result['Sex'] = group['Sex'].iloc[0]
    result['Weight'] = group['Weight'].iloc[0]
    result['Height'] = group['Height'].iloc[0]
    result['Humidity'] = group['Humidity'].iloc[0]
    result['Temperature'] = group['Temperature'].iloc[0]
    
    # 插值各个生理变量
    for col in ['Speed', 'HR', 'VO2', 'VCO2', 'RR', 'VE']:
        if col not in group.columns:
            continue
        
        # 尝试转换为数值
        try:
            y_vals = pd.to_numeric(group[col], errors='coerce').values
            # 检查是否有有效的数据
            valid_idx = ~np.isnan(y_vals)
            if valid_idx.sum() < 5:
                result[col] = None
                continue
            
            # 用有效数据进行插值
            f_interp = interp1d(
                time_vals[valid_idx],
                y_vals[valid_idx],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            interp_vals = f_interp(target_grid)
            # 确保没有 NaN
            if np.any(np.isnan(interp_vals)):
                result[col] = None
            else:
                result[col] = interp_vals.tolist()
        except Exception as e:
            print(f"  ⚠ 警告: ID_test={result['ID_test']}, 列 {col} 插值失败: {e}")
            result[col] = None
    
    result['time_grid'] = target_grid.tolist()
    
    return result

# 处理所有 workouts
processed_workouts = []
skipped_count = 0

for idx, (wid, group) in enumerate(grouped):
    if (idx + 1) % 100 == 0:
        print(f"  已处理: {idx + 1}/{grouped.ngroups}")
    
    processed = process_workout(group)
    if processed is not None:
        processed_workouts.append(processed)
    else:
        skipped_count += 1

print(f"\n✓ 成功处理: {len(processed_workouts)} 个 workouts")
print(f"✗ 跳过: {skipped_count} 个 workouts (数据质量问题)")

# ============================================================================
# 5. 转成 DataFrame，并保存
# ============================================================================
print("\n" + "=" * 80)
print("数据保存")
print("=" * 80)

df_processed = pd.DataFrame(processed_workouts)

print(f"\n处理后数据形状: {df_processed.shape}")
print(f"\n列名:\n  {list(df_processed.columns)}")

# 检查是否所有必要列都有
required_cols = ['ID_test', 'ID', 'Age', 'Sex', 'Weight', 'Height', 
                 'Temperature', 'Humidity', 'time_grid', 'HR', 'Speed']
missing_cols = [c for c in required_cols if c not in df_processed.columns]
if missing_cols:
    print(f"\n⚠ 缺少的列: {missing_cols}")

# 保存为 pickle 格式（支持嵌套数据）
output_pkl = 'data/processed_data.pkl'
with open(output_pkl, 'wb') as f:
    pickle.dump(df_processed, f)
print(f"✓ 已保存 (pickle): {output_pkl}")

# 同时保存元数据为 CSV（不含列表列）
metadata_cols = ['ID_test', 'ID', 'Age', 'Sex', 'Weight', 'Height', 
                 'Humidity', 'Temperature', 'duration', 'n_measurements', 'n_interpolated_points']
df_metadata = df_processed[metadata_cols].copy()
output_csv = 'data/processed_metadata.csv'
df_metadata.to_csv(output_csv, index=False)
print(f"✓ 已保存 (CSV 元数据): {output_csv}")

# ============================================================================
# 6. 生成数据质量报告
# ============================================================================
print("\n" + "=" * 80)
print("数据质量报告")
print("=" * 80)

report_lines = []

report_lines.append("=" * 80)
report_lines.append("STAGE A 数据处理报告")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append("输入数据统计:")
report_lines.append(f"  - subject-info.csv: {subject_info.shape[0]} 行")
report_lines.append(f"  - test_measure.csv: {test_measure.shape[0]} 行")
report_lines.append(f"  - 独特 subjects: {n_subjects}")
report_lines.append(f"  - 独特 workouts: {n_workouts}")
report_lines.append("")

report_lines.append("输出数据统计:")
report_lines.append(f"  - 处理成功: {len(processed_workouts)} 个 workouts")
report_lines.append(f"  - 处理失败/跳过: {skipped_count} 个 workouts")
report_lines.append(f"  - 成功率: {100*len(processed_workouts)/n_workouts:.1f}%")
report_lines.append("")

# 统计各列的非空情况
report_lines.append("各列非空统计:")
for col in ['Speed', 'HR', 'VO2', 'VCO2', 'RR', 'VE']:
    non_null = df_processed[col].notna().sum()
    report_lines.append(f"  - {col}: {non_null}/{len(df_processed)} 个 workouts")
report_lines.append("")

# 统计序列长度
if 'time_grid' in df_processed.columns:
    lengths = df_processed['time_grid'].apply(len)
    report_lines.append("时间序列长度统计:")
    report_lines.append(f"  - 最小: {lengths.min()}")
    report_lines.append(f"  - 最大: {lengths.max()}")
    report_lines.append(f"  - 平均: {lengths.mean():.1f}")
    report_lines.append(f"  - 中位数: {lengths.median():.1f}")
    report_lines.append("")

# 个体特征统计
report_lines.append("个体特征统计:")
report_lines.append(f"  - 年龄范围: {df_processed['Age'].min():.1f} - {df_processed['Age'].max():.1f}")
report_lines.append(f"  - 体重范围: {df_processed['Weight'].min():.1f} - {df_processed['Weight'].max():.1f}")
report_lines.append(f"  - 身高范围: {df_processed['Height'].min():.1f} - {df_processed['Height'].max():.1f}")
report_lines.append("")

report_lines.append("输出文件:")
report_lines.append(f"  - {output_pkl}")
report_lines.append(f"  - {output_csv}")
report_lines.append("")

report_text = "\n".join(report_lines)
print(report_text)

# 保存报告
report_path = 'data/data_quality_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"✓ 报告已保存: {report_path}")

# ============================================================================
# 7. 快速预览
# ============================================================================
print("\n" + "=" * 80)
print("处理后数据快速预览")
print("=" * 80)

print(f"\n前5行的元数据:")
print(df_processed[['ID_test', 'ID', 'Age', 'Sex', 'Weight', 'Height', 
                     'Temperature', 'Humidity', 'duration', 'n_measurements']].head())

print(f"\n某个 workout 的数据结构 (第1行):")
row1 = df_processed.iloc[0]
print(f"  ID_test: {row1['ID_test']}")
print(f"  time_grid 长度: {len(row1['time_grid']) if row1['time_grid'] else 'N/A'}")
if row1['HR'] is not None:
    hr_list = row1['HR'] if isinstance(row1['HR'], list) else []
    print(f"  HR 长度: {len(hr_list)}, 范围: [{min(hr_list):.1f}, {max(hr_list):.1f}]")
if row1['Speed'] is not None:
    speed_list = row1['Speed'] if isinstance(row1['Speed'], list) else []
    print(f"  Speed 长度: {len(speed_list)}, 范围: [{min(speed_list):.2f}, {max(speed_list):.2f}]")

print("\n" + "=" * 80)
print("✓ STAGE A 完成!")
print("=" * 80)
