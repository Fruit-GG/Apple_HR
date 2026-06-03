#!/usr/bin/env python3
"""
Stage B: Data Preprocessing
================================
对处理后的数据进行特征工程和标准化。

输入：
  - data/processed_data.pkl：阶段 A 的输出

输出：
  - data/preprocessed_data.pkl：标准化后的数据
  - data/preprocessing_stats.pkl：标准化参数（用于测试集）
  - data/preprocessing_report.txt：详细报告
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 加载数据
# ============================================================================
print("=" * 80)
print("STAGE B: 数据预处理 - 加载阶段 A 的输出")
print("=" * 80)

with open('data/processed_data.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"\n✓ 加载完成: {df.shape[0]} 个 workouts")
print(f"✓ 列名: {list(df.columns)}")

# ============================================================================
# 2. 初步数据检查与过滤
# ============================================================================
print("\n" + "=" * 80)
print("数据质量过滤")
print("=" * 80)

initial_count = len(df)

# 过滤 1: 移除 HR 为 None 的 workout
df = df[df['HR'].notna()].reset_index(drop=True)
print(f"\n✓ 过滤 HR=None: {initial_count} → {len(df)} ({len(df)-initial_count} 个跳过)")

# 过滤 2: 移除 Speed 为 None 的 workout
initial_count = len(df)
df = df[df['Speed'].notna()].reset_index(drop=True)
print(f"✓ 过滤 Speed=None: {initial_count} → {len(df)} ({len(df)-initial_count} 个跳过)")

# 过滤 3: 过滤太短的时间序列（少于 10 个点）
initial_count = len(df)
df = df[df['HR'].apply(len) >= 10].reset_index(drop=True)
print(f"✓ 过滤序列长度<10: {initial_count} → {len(df)} ({len(df)-initial_count} 个跳过)")

# 过滤 4: 检查 HR 值的合理性（心率范围通常 30-220 BPM）
initial_count = len(df)
def check_hr_range(hr_list):
    if hr_list is None:
        return False
    hr_array = np.array(hr_list)
    return (hr_array.min() >= 30) and (hr_array.max() <= 220)

df = df[df['HR'].apply(check_hr_range)].reset_index(drop=True)
print(f"✓ 过滤 HR 范围异常: {initial_count} → {len(df)} ({len(df)-initial_count} 个跳过)")

# 过滤 5: 检查 Speed 值的合理性（跑步速度通常 2-25 km/h）
initial_count = len(df)
def check_speed_range(speed_list):
    if speed_list is None:
        return False
    speed_array = np.array(speed_list)
    return (speed_array.min() > 0) and (speed_array.max() <= 30)

df = df[df['Speed'].apply(check_speed_range)].reset_index(drop=True)
print(f"✓ 过滤 Speed 范围异常: {initial_count} → {len(df)} ({len(df)-initial_count} 个跳过)")

print(f"\n最终保留: {len(df)}/{initial_count} 个 workouts")

# ============================================================================
# 3. 特征工程与标准化参数计算
# ============================================================================
print("\n" + "=" * 80)
print("特征工程与标准化")
print("=" * 80)

# ========== 3.1: 个体特征的标准化 ==========
print("\n1. 个体特征标准化...")

# Age, Weight, Height 使用标准化 (z-score)
scaler_continuous = StandardScaler()
continuous_features = ['Age', 'Weight', 'Height']

# 计算统计量（用于保存和应用到测试集）
stats = {}
for feat in continuous_features:
    stats[f'{feat}_mean'] = df[feat].mean()
    stats[f'{feat}_std'] = df[feat].std()
    print(f"  {feat}: 均值={stats[f'{feat}_mean']:.2f}, 标准差={stats[f'{feat}_std']:.2f}")

# ========== 3.2: 心率标准化 ==========
print("\n2. 心率标准化...")
# 按论文方式：(HR - mean_HR) / std_HR
# 通常跑步数据的 HR 均值 ~120-140, 标准差 ~20-30

def normalize_hr(hr_list):
    """按全局统计进行标准化"""
    if hr_list is None:
        return None
    hr_array = np.array(hr_list)
    # 使用全局统计量
    hr_norm = (hr_array - stats['HR_mean']) / (stats['HR_std'] + 1e-8)
    return hr_norm.tolist()

# 计算 HR 的全局统计
all_hr_values = []
for hr_list in df['HR']:
    if hr_list is not None:
        all_hr_values.extend(hr_list)
all_hr_values = np.array(all_hr_values)
stats['HR_mean'] = all_hr_values.mean()
stats['HR_std'] = all_hr_values.std()
stats['HR_min'] = all_hr_values.min()
stats['HR_max'] = all_hr_values.max()

print(f"  HR 全局统计:")
print(f"    均值: {stats['HR_mean']:.2f} BPM")
print(f"    标准差: {stats['HR_std']:.2f} BPM")
print(f"    范围: [{stats['HR_min']:.2f}, {stats['HR_max']:.2f}] BPM")

# ========== 3.3: Speed 归一化 ==========
print("\n3. 运动强度 (Speed) 归一化...")
# Speed 使用 Min-Max 归一化 (0-1)

def normalize_speed(speed_list):
    """按全局统计进行 min-max 归一化"""
    if speed_list is None:
        return None
    speed_array = np.array(speed_list)
    # 使用全局最小最大值
    speed_norm = (speed_array - stats['Speed_min']) / (stats['Speed_max'] - stats['Speed_min'] + 1e-8)
    # 限制在 [0, 1]
    speed_norm = np.clip(speed_norm, 0, 1)
    return speed_norm.tolist()

all_speed_values = []
for speed_list in df['Speed']:
    if speed_list is not None:
        all_speed_values.extend(speed_list)
all_speed_values = np.array(all_speed_values)
stats['Speed_mean'] = all_speed_values.mean()
stats['Speed_std'] = all_speed_values.std()
stats['Speed_min'] = all_speed_values.min()
stats['Speed_max'] = all_speed_values.max()

print(f"  Speed 全局统计:")
print(f"    均值: {stats['Speed_mean']:.2f}")
print(f"    标准差: {stats['Speed_std']:.2f}")
print(f"    范围: [{stats['Speed_min']:.2f}, {stats['Speed_max']:.2f}]")

# ========== 3.4: VO2 等辅助变量的处理 ==========
print("\n4. 辅助生理变量处理...")

def normalize_physiological(col_name, data_values):
    """
    对生理变量进行标准化
    """
    all_values = []
    for val_list in data_values:
        if val_list is not None:
            all_values.extend(val_list)
    
    if not all_values:
        return None, None, None, None
    
    all_values = np.array(all_values)
    mean_val = all_values.mean()
    std_val = all_values.std()
    min_val = all_values.min()
    max_val = all_values.max()
    
    print(f"  {col_name}: 均值={mean_val:.2f}, 标准差={std_val:.2f}, " +
          f"范围=[{min_val:.2f}, {max_val:.2f}]")
    
    return mean_val, std_val, min_val, max_val

for col in ['VO2', 'VCO2', 'RR', 'VE']:
    # 仅处理非空的
    valid_data = df[df[col].notna()][col]
    if len(valid_data) > 0:
        mean_val, std_val, min_val, max_val = normalize_physiological(col, valid_data)
        stats[f'{col}_mean'] = mean_val
        stats[f'{col}_std'] = std_val
        stats[f'{col}_min'] = min_val
        stats[f'{col}_max'] = max_val

# ========== 3.5: Sex 编码 ==========
print("\n5. 性别编码...")
# Sex 已经是 0/1，但可以标准化一下（可选）
stats['Sex_values'] = df['Sex'].unique().tolist()
print(f"  Sex 唯一值: {stats['Sex_values']}")

# ============================================================================
# 4. 应用规范化到所有数据
# ============================================================================
print("\n" + "=" * 80)
print("应用标准化规则")
print("=" * 80)

df_normalized = df.copy()

print("\n应用中...")

# 标准化个体特征
for feat in continuous_features:
    df_normalized[f'{feat}_norm'] = (df[feat] - stats[f'{feat}_mean']) / (stats[f'{feat}_std'] + 1e-8)

# 标准化 HR
df_normalized['HR_normalized'] = df['HR'].apply(normalize_hr)

# 归一化 Speed
df_normalized['Speed_normalized'] = df['Speed'].apply(normalize_speed)

# 标准化辅助变量
for col in ['VO2', 'VCO2', 'RR', 'VE']:
    def normalize_col(col_list, col_name):
        if col_list is None:
            return None
        col_array = np.array(col_list)
        if col_name in stats and stats[f'{col_name}_std'] is not None:
            col_norm = (col_array - stats[f'{col_name}_mean']) / (stats[f'{col_name}_std'] + 1e-8)
            return col_norm.tolist()
        return None
    
    df_normalized[f'{col}_normalized'] = df[col].apply(lambda x: normalize_col(x, col))

print("✓ 规范化完成")

# ============================================================================
# 5. 派生特征
# ============================================================================
print("\n" + "=" * 80)
print("派生特征生成")
print("=" * 80)

# 心率变异性 (HR Variability)
def calculate_hr_variability(hr_list):
    if hr_list is None or len(hr_list) < 2:
        return None
    hr_array = np.array(hr_list)
    return float(np.std(hr_array))

# 心率平均值
def calculate_mean_hr(hr_list):
    if hr_list is None:
        return None
    return float(np.mean(hr_list))

# Speed 平均值
def calculate_mean_speed(speed_list):
    if speed_list is None:
        return None
    return float(np.mean(speed_list))

# 心率上升率 (HR slope)
def calculate_hr_slope(hr_list):
    if hr_list is None or len(hr_list) < 2:
        return None
    hr_array = np.array(hr_list)
    # 线性回归斜率
    x = np.arange(len(hr_array))
    slope = np.polyfit(x, hr_array, 1)[0]
    return float(slope)

df_normalized['HR_var'] = df['HR'].apply(calculate_hr_variability)
df_normalized['HR_mean'] = df['HR'].apply(calculate_mean_hr)
df_normalized['Speed_mean'] = df['Speed'].apply(calculate_mean_speed)
df_normalized['HR_slope'] = df['HR'].apply(calculate_hr_slope)

# 标准化派生特征
for feat in ['HR_var', 'HR_mean', 'HR_slope', 'Speed_mean']:
    feat_values = df_normalized[feat].dropna().values
    if len(feat_values) > 0:
        feat_mean = feat_values.mean()
        feat_std = feat_values.std()
        stats[f'{feat}_mean'] = feat_mean
        stats[f'{feat}_std'] = feat_std
        df_normalized[f'{feat}_norm'] = (df_normalized[feat] - feat_mean) / (feat_std + 1e-8)

print("✓ 派生特征:")
print(f"  - HR_var: 心率变异性")
print(f"  - HR_mean: 平均心率")
print(f"  - Speed_mean: 平均运动强度")
print(f"  - HR_slope: 心率上升趋势")

# ============================================================================
# 6. 保存数据
# ============================================================================
print("\n" + "=" * 80)
print("保存预处理后的数据")
print("=" * 80)

# 保存完整预处理数据
output_pkl = 'data/preprocessed_data.pkl'
with open(output_pkl, 'wb') as f:
    pickle.dump(df_normalized, f)
print(f"\n✓ 已保存: {output_pkl}")

# 保存统计参数（用于测试集）
stats_pkl = 'data/preprocessing_stats.pkl'
with open(stats_pkl, 'wb') as f:
    pickle.dump(stats, f)
print(f"✓ 已保存（统计参数）: {stats_pkl}")

# 保存元数据为 CSV
metadata_cols = ['ID_test', 'ID', 'Age', 'Weight', 'Height', 'Sex', 
                 'Humidity', 'Temperature', 'duration', 'n_measurements']
df_normalized_meta = df_normalized[metadata_cols].copy()
# 加上派生特征
for feat in ['HR_var', 'HR_mean', 'Speed_mean', 'HR_slope', 'duration']:
    if feat in df_normalized.columns:
        df_normalized_meta[feat] = df_normalized[feat]

output_csv = 'data/preprocessed_metadata.csv'
df_normalized_meta.to_csv(output_csv, index=False)
print(f"✓ 已保存（元数据 CSV）: {output_csv}")

# ============================================================================
# 7. 生成详细报告
# ============================================================================
print("\n" + "=" * 80)
print("生成预处理报告")
print("=" * 80)

report_lines = []

report_lines.append("=" * 80)
report_lines.append("STAGE B 数据预处理报告")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append("1. 数据过滤统计")
report_lines.append("-" * 80)
report_lines.append(f"  初始 workouts (来自 Stage A): {initial_count + sum([len(df) if i == 0 else 0 for i in range(1)])}")
report_lines.append(f"  最终保留: {len(df_normalized)} workouts")
report_lines.append(f"  过滤率: {(1 - len(df_normalized)/initial_count)*100:.1f}%")
report_lines.append("")

report_lines.append("2. 个体特征统计")
report_lines.append("-" * 80)
for feat in continuous_features:
    report_lines.append(f"  {feat}:")
    report_lines.append(f"    原始值 - 均值: {df[feat].mean():.2f}, 标准差: {df[feat].std():.2f}")
    report_lines.append(f"    标准化参数: μ={stats[f'{feat}_mean']:.2f}, σ={stats[f'{feat}_std']:.2f}")
report_lines.append(f"  Sex: {df['Sex'].value_counts().to_dict()}")
report_lines.append("")

report_lines.append("3. 时间序列变量标准化")
report_lines.append("-" * 80)
report_lines.append(f"  HR (心率):")
report_lines.append(f"    原始值 - 均值: {stats['HR_mean']:.2f} BPM, 标准差: {stats['HR_std']:.2f} BPM")
report_lines.append(f"    范围: [{stats['HR_min']:.2f}, {stats['HR_max']:.2f}] BPM")
report_lines.append(f"  Speed (运动强度):")
report_lines.append(f"    原始值 - 均值: {stats['Speed_mean']:.2f}, 标准差: {stats['Speed_std']:.2f}")
report_lines.append(f"    范围: [{stats['Speed_min']:.2f}, {stats['Speed_max']:.2f}]")
report_lines.append("")

report_lines.append("4. 派生特征统计")
report_lines.append("-" * 80)
for feat in ['HR_var', 'HR_mean', 'Speed_mean', 'HR_slope']:
    if feat in df_normalized.columns:
        valid_vals = df_normalized[feat].dropna().values
        if len(valid_vals) > 0:
            report_lines.append(f"  {feat}:")
            report_lines.append(f"    均值: {valid_vals.mean():.2f}, 标准差: {valid_vals.std():.2f}")
            report_lines.append(f"    范围: [{valid_vals.min():.2f}, {valid_vals.max():.2f}]")
report_lines.append("")

report_lines.append("5. 输出文件")
report_lines.append("-" * 80)
report_lines.append(f"  - {output_pkl}: 完整预处理数据（包含所有时间序列）")
report_lines.append(f"  - {stats_pkl}: 标准化统计参数（用于测试集）")
report_lines.append(f"  - {output_csv}: 元数据 CSV（易于查看）")
report_lines.append("")

report_lines.append("6. 使用说明")
report_lines.append("-" * 80)
report_lines.append("  在测试或预测时，使用 preprocessing_stats.pkl 中的参数对新数据进行相同的标准化。")
report_lines.append("  标准化公式 (对于连续特征):")
report_lines.append("    x_norm = (x - μ) / σ")
report_lines.append("")

report_text = "\n".join(report_lines)
print(report_text)

# 保存报告
report_path = 'data/preprocessing_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"✓ 报告已保存: {report_path}")

# ============================================================================
# 8. 快速预览
# ============================================================================
print("\n" + "=" * 80)
print("预处理后数据快速预览")
print("=" * 80)

print(f"\n列名 ({len(df_normalized.columns)} 列):")
print(f"  原始列: {[c for c in df_normalized.columns if not c.endswith('_norm') and not c.endswith('_normalized')]}")
print(f"  标准化列: {[c for c in df_normalized.columns if c.endswith('_norm') or c.endswith('_normalized')]}")

print(f"\n第一个 workout 预览 (ID_test={df_normalized.iloc[0]['ID_test']}):")
row = df_normalized.iloc[0]
print(f"  Age: {row['Age']:.1f} (标准化: {row['Age_norm']:.3f})")
print(f"  Weight: {row['Weight']:.1f} (标准化: {row['Weight_norm']:.3f})")
print(f"  Height: {row['Height']:.1f} (标准化: {row['Height_norm']:.3f})")
print(f"  Sex: {row['Sex']}")
print(f"  HR_mean: {row['HR_mean']:.2f} (标准化: {row['HR_mean_norm']:.3f})")
print(f"  Speed_mean: {row['Speed_mean']:.2f}")
if row['HR_normalized'] is not None:
    print(f"  HR_normalized 样本 (前 5 个): {row['HR_normalized'][:5]}")
if row['Speed_normalized'] is not None:
    print(f"  Speed_normalized 样本 (前 5 个): {row['Speed_normalized'][:5]}")

print("\n" + "=" * 80)
print("✓ STAGE B 完成!")
print("=" * 80)
