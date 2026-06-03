#!/usr/bin/env python3
"""
Stage C: Data Splitting
================================
将预处理数据按 subject-level 和时间顺序分成训练/验证/测试集。

输入：
  - data/preprocessed_data.pkl：阶段 B 的输出

输出：
  - data/train_data.pkl：训练集
  - data/val_data.pkl：验证集
  - data/test_data.pkl：测试集
  - data/split_summary.pkl：划分统计信息
  - data/subject_split_info.csv：各 subject 的划分信息
  - data/split_report.txt：详细报告
"""

import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 加载数据
# ============================================================================
print("=" * 80)
print("STAGE C: 数据划分 - 加载预处理数据")
print("=" * 80)

with open('data/preprocessed_data.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"\n✓ 加载完成: {len(df)} 个 workouts")

# ============================================================================
# 2. 按 Subject 分组并排序
# ============================================================================
print("\n" + "=" * 80)
print("按 Subject 分组与排序")
print("=" * 80)

# 按 ID (subject) 分组
grouped_by_subject = df.groupby('ID')
n_subjects = len(grouped_by_subject)

print(f"\n✓ 独特 subjects: {n_subjects}")

# 对每个 subject，按时间顺序排列 workouts
# 由于没有明确的时间戳，我们按 ID_test 的数字顺序排列
subject_data = {}
subject_workout_counts = {}

for subject_id, group in grouped_by_subject:
    # 按 ID_test 的数字部分排序（ID_test 格式如 "100_1", "100_2" 等）
    group = group.copy()
    
    # 解析 ID_test 中的数字用于排序
    def extract_workout_number(id_test_str):
        # ID_test 格式: "{ID}_{数字}"
        parts = str(id_test_str).split('_')
        if len(parts) > 1:
            try:
                return int(parts[-1])
            except:
                return 0
        return 0
    
    group['_workout_order'] = group['ID_test'].apply(extract_workout_number)
    group = group.sort_values('_workout_order').reset_index(drop=True)
    group = group.drop('_workout_order', axis=1)
    
    subject_data[subject_id] = group
    subject_workout_counts[subject_id] = len(group)

print("\nSubject 级工作量统计:")
counts = list(subject_workout_counts.values())
print(f"  平均每个 subject 的 workouts: {np.mean(counts):.1f}")
print(f"  中位数: {np.median(counts):.1f}")
print(f"  范围: [{min(counts)}, {max(counts)}]")

# ============================================================================
# 3. 混合策略划分：多 workout subjects 避免混合，单 workout subjects 随机划分
# ============================================================================
print("\n" + "=" * 80)
print("数据划分策略")
print("=" * 80)

# 划分比例：60% 训练，20% 验证，20% 测试
train_ratio = 0.60
val_ratio = 0.20
test_ratio = 0.20

print(f"\n划分比例:")
print(f"  训练集: {train_ratio*100:.0f}%")
print(f"  验证集: {val_ratio*100:.0f}%")
print(f"  测试集: {test_ratio*100:.0f}%")

# 识别多 workout 和单 workout 的 subjects
multi_workout_subjects = {sid: group_df for sid, group_df in subject_data.items() 
                          if len(group_df) > 1}
single_workout_subjects = {sid: group_df for sid, group_df in subject_data.items() 
                           if len(group_df) == 1}

print(f"\nSubject 分类:")
print(f"  多 workout subjects (>1): {len(multi_workout_subjects)} " +
      f"({sum(len(g) for g in multi_workout_subjects.values())} workouts)")
print(f"  单 workout subjects (=1): {len(single_workout_subjects)} " +
      f"({len(single_workout_subjects)} workouts)")

train_indices = []
val_indices = []
test_indices = []

subject_split_info = []

# 处理多 workout subjects：按 chronological 分割整个 subject
print(f"\n按 subject 划分中...")

for subject_id, group_df in multi_workout_subjects.items():
    n_workouts = len(group_df)
    
    # 按比例计算分割点
    train_end = int(n_workouts * train_ratio)
    val_end = train_end + int(n_workouts * val_ratio)
    
    # 获取该 subject 的索引
    subject_indices = group_df.index.tolist()
    
    # 分割
    train_idx = subject_indices[:train_end]
    val_idx = subject_indices[train_end:val_end]
    test_idx = subject_indices[val_end:]
    
    train_indices.extend(train_idx)
    val_indices.extend(val_idx)
    test_indices.extend(test_idx)
    
    # 记录划分信息
    subject_split_info.append({
        'ID': subject_id,
        'total_workouts': n_workouts,
        'train_count': len(train_idx),
        'val_count': len(val_idx),
        'test_count': len(test_idx),
        'train_pct': f"{len(train_idx)/n_workouts*100:.1f}%",
        'val_pct': f"{len(val_idx)/n_workouts*100:.1f}%",
        'test_pct': f"{len(test_idx)/n_workouts*100:.1f}%"
    })

# 处理单 workout subjects：随机分配
single_workout_indices = [subject_data[sid].index[0] for sid in single_workout_subjects.keys()]
np.random.shuffle(single_workout_indices)

n_single = len(single_workout_indices)
train_end = int(n_single * train_ratio)
val_end = train_end + int(n_single * val_ratio)

train_indices.extend(single_workout_indices[:train_end])
val_indices.extend(single_workout_indices[train_end:val_end])
test_indices.extend(single_workout_indices[val_end:])

# 记录单 workout subjects 的划分信息
for i, subject_id in enumerate(single_workout_subjects.keys()):
    if i < train_end:
        split = 'train'
    elif i < val_end:
        split = 'val'
    else:
        split = 'test'
    
    subject_split_info.append({
        'ID': subject_id,
        'total_workouts': 1,
        'train_count': 1 if split == 'train' else 0,
        'val_count': 1 if split == 'val' else 0,
        'test_count': 1 if split == 'test' else 0,
        'train_pct': '100.0%' if split == 'train' else '0.0%',
        'val_pct': '100.0%' if split == 'val' else '0.0%',
        'test_pct': '100.0%' if split == 'test' else '0.0%'
    })

print(f"✓ 划分完成")

# ============================================================================
# 4. 生成训练/验证/测试集
# ============================================================================
print("\n" + "=" * 80)
print("生成数据集")
print("=" * 80)

train_data = df.loc[train_indices].reset_index(drop=True)
val_data = df.loc[val_indices].reset_index(drop=True)
test_data = df.loc[test_indices].reset_index(drop=True)

print(f"\n✓ 训练集: {len(train_data)} workouts ({len(train_data)/len(df)*100:.1f}%)")
print(f"✓ 验证集: {len(val_data)} workouts ({len(val_data)/len(df)*100:.1f}%)")
print(f"✓ 测试集: {len(test_data)} workouts ({len(test_data)/len(df)*100:.1f}%)")

# ============================================================================
# 5. 检查数据集统计
# ============================================================================
print("\n" + "=" * 80)
print("数据集统计")
print("=" * 80)

def compute_dataset_stats(data, name):
    print(f"\n{name}:")
    print(f"  样本数: {len(data)}")
    print(f"  独特 subjects: {data['ID'].nunique()}")
    
    # Age 统计
    if 'Age' in data.columns:
        age_stats = data['Age'].agg(['mean', 'std', 'min', 'max'])
        print(f"  Age: μ={age_stats['mean']:.1f}, σ={age_stats['std']:.1f}, " +
              f"范围=[{age_stats['min']:.1f}, {age_stats['max']:.1f}]")
    
    # HR_mean 统计
    if 'HR_mean' in data.columns:
        hr_stats = data['HR_mean'].agg(['mean', 'std', 'min', 'max'])
        print(f"  HR_mean: μ={hr_stats['mean']:.1f}, σ={hr_stats['std']:.1f}, " +
              f"范围=[{hr_stats['min']:.1f}, {hr_stats['max']:.1f}]")
    
    # Speed_mean 统计
    if 'Speed_mean' in data.columns:
        speed_stats = data['Speed_mean'].agg(['mean', 'std', 'min', 'max'])
        print(f"  Speed_mean: μ={speed_stats['mean']:.2f}, σ={speed_stats['std']:.2f}, " +
              f"范围=[{speed_stats['min']:.2f}, {speed_stats['max']:.2f}]")
    
    # Sex 分布
    if 'Sex' in data.columns:
        sex_counts = data['Sex'].value_counts()
        print(f"  Sex: {dict(sex_counts)}")

compute_dataset_stats(train_data, "训练集")
compute_dataset_stats(val_data, "验证集")
compute_dataset_stats(test_data, "测试集")

# ============================================================================
# 6. 数据泄露检查
# ============================================================================
print("\n" + "=" * 80)
print("数据泄露检查")
print("=" * 80)

train_subjects = set(train_data['ID'].unique())
val_subjects = set(val_data['ID'].unique())
test_subjects = set(test_data['ID'].unique())

train_val_overlap = train_subjects & val_subjects
val_test_overlap = val_subjects & test_subjects
train_test_overlap = train_subjects & test_subjects

print("\n检查结果:")
if len(train_val_overlap) == 0:
    print("  ✓ 训练集和验证集无 subject 重叠")
else:
    print(f"  ✗ 警告: 训练集和验证集有 {len(train_val_overlap)} 个 subject 重叠")

if len(val_test_overlap) == 0:
    print("  ✓ 验证集和测试集无 subject 重叠")
else:
    print(f"  ✗ 警告: 验证集和测试集有 {len(val_test_overlap)} 个 subject 重叠")

if len(train_test_overlap) == 0:
    print("  ✓ 训练集和测试集无 subject 重叠")
else:
    print(f"  ✗ 警告: 训练集和测试集有 {len(train_test_overlap)} 个 subject 重叠")

# ============================================================================
# 7. 保存数据集
# ============================================================================
print("\n" + "=" * 80)
print("保存数据集")
print("=" * 80)

# 保存训练/验证/测试集
for dataset, filename in [
    (train_data, 'data/train_data.pkl'),
    (val_data, 'data/val_data.pkl'),
    (test_data, 'data/test_data.pkl'),
]:
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"✓ 已保存: {filename}")

# 保存划分统计信息
split_summary = {
    'train_count': len(train_data),
    'val_count': len(val_data),
    'test_count': len(test_data),
    'train_subjects': len(train_subjects),
    'val_subjects': len(val_subjects),
    'test_subjects': len(test_subjects),
    'total_subjects': n_subjects,
    'train_ratio': len(train_data) / len(df),
    'val_ratio': len(val_data) / len(df),
    'test_ratio': len(test_data) / len(df),
    'split_strategy': 'subject-level chronological split',
}

with open('data/split_summary.pkl', 'wb') as f:
    pickle.dump(split_summary, f)
print(f"✓ 已保存: data/split_summary.pkl")

# 保存 subject 级划分信息
df_split_info = pd.DataFrame(subject_split_info)
df_split_info.to_csv('data/subject_split_info.csv', index=False)
print(f"✓ 已保存: data/subject_split_info.csv")

# ============================================================================
# 8. 生成详细报告
# ============================================================================
print("\n" + "=" * 80)
print("生成划分报告")
print("=" * 80)

report_lines = []

report_lines.append("=" * 80)
report_lines.append("STAGE C 数据划分报告")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append("1. 划分策略")
report_lines.append("-" * 80)
report_lines.append("  方法: 混合 Subject-level Chronological 和随机划分")
report_lines.append("  说明:")
report_lines.append("    - 对于有多个 workouts 的 subjects:")
report_lines.append("      * 同一个 subject 的 workouts 不会同时出现在不同集合中（避免数据泄露）")
report_lines.append("      * 按时间顺序分割")
report_lines.append("    - 对于只有 1 个 workout 的 subjects:")
report_lines.append("      * 随机分配到三个集合中")
report_lines.append("")
report_lines.append(f"  目标比例: 训练 {train_ratio*100:.0f}% / 验证 {val_ratio*100:.0f}% / 测试 {test_ratio*100:.0f}%")
report_lines.append("")

report_lines.append("2. 划分结果")
report_lines.append("-" * 80)
report_lines.append(f"  总样本数: {len(df)}")
report_lines.append(f"  总 subjects: {n_subjects}")
report_lines.append("")
report_lines.append(f"  训练集: {len(train_data)} workouts ({len(train_data)/len(df)*100:.1f}%), " +
                   f"{len(train_subjects)} subjects")
report_lines.append(f"  验证集: {len(val_data)} workouts ({len(val_data)/len(df)*100:.1f}%), " +
                   f"{len(val_subjects)} subjects")
report_lines.append(f"  测试集: {len(test_data)} workouts ({len(test_data)/len(df)*100:.1f}%), " +
                   f"{len(test_subjects)} subjects")
report_lines.append("")

report_lines.append("3. 数据泄露检查")
report_lines.append("-" * 80)
report_lines.append(f"  训练-验证 subject 重叠: {len(train_val_overlap)} " +
                   ("✓" if len(train_val_overlap) == 0 else "✗"))
report_lines.append(f"  验证-测试 subject 重叠: {len(val_test_overlap)} " +
                   ("✓" if len(val_test_overlap) == 0 else "✗"))
report_lines.append(f"  训练-测试 subject 重叠: {len(train_test_overlap)} " +
                   ("✓" if len(train_test_overlap) == 0 else "✗"))
report_lines.append("")

report_lines.append("4. 特征统计 (训练集)")
report_lines.append("-" * 80)
if 'Age' in train_data.columns:
    age_stats = train_data['Age'].agg(['mean', 'std', 'min', 'max'])
    report_lines.append(f"  Age: μ={age_stats['mean']:.1f}, σ={age_stats['std']:.1f}, " +
                       f"范围=[{age_stats['min']:.1f}, {age_stats['max']:.1f}]")

if 'HR_mean' in train_data.columns:
    hr_stats = train_data['HR_mean'].agg(['mean', 'std', 'min', 'max'])
    report_lines.append(f"  HR_mean: μ={hr_stats['mean']:.1f}, σ={hr_stats['std']:.1f}, " +
                       f"范围=[{hr_stats['min']:.1f}, {hr_stats['max']:.1f}]")

if 'Speed_mean' in train_data.columns:
    speed_stats = train_data['Speed_mean'].agg(['mean', 'std', 'min', 'max'])
    report_lines.append(f"  Speed_mean: μ={speed_stats['mean']:.2f}, σ={speed_stats['std']:.2f}, " +
                       f"范围=[{speed_stats['min']:.2f}, {speed_stats['max']:.2f}]")

if 'Sex' in train_data.columns:
    sex_counts = train_data['Sex'].value_counts()
    report_lines.append(f"  Sex: {dict(sex_counts)}")

report_lines.append("")

report_lines.append("5. 输出文件")
report_lines.append("-" * 80)
report_lines.append("  - data/train_data.pkl: 训练数据")
report_lines.append("  - data/val_data.pkl: 验证数据")
report_lines.append("  - data/test_data.pkl: 测试数据")
report_lines.append("  - data/split_summary.pkl: 划分摘要（用于记录）")
report_lines.append("  - data/subject_split_info.csv: 各 subject 的划分详情")
report_lines.append("")

report_lines.append("6. 使用说明")
report_lines.append("-" * 80)
report_lines.append("  训练模型时:")
report_lines.append("    - 使用 train_data.pkl 训练")
report_lines.append("    - 使用 val_data.pkl 验证（早停、超参调优等）")
report_lines.append("    - 使用 test_data.pkl 最终评估")
report_lines.append("")
report_lines.append("  加载方式:")
report_lines.append("    import pickle")
report_lines.append("    with open('data/train_data.pkl', 'rb') as f:")
report_lines.append("        train_data = pickle.load(f)")
report_lines.append("")

report_text = "\n".join(report_lines)
print(report_text)

# 保存报告
report_path = 'data/split_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"✓ 报告已保存: {report_path}")

# ============================================================================
# 9. 快速预览
# ============================================================================
print("\n" + "=" * 80)
print("数据集预览")
print("=" * 80)

print("\n前 5 个训练样本:")
print(train_data[['ID_test', 'ID', 'Age', 'Sex', 'HR_mean', 'Speed_mean']].head())

print("\n验证集样本数量分布 (按 subject):")
val_subject_counts = val_data['ID'].value_counts().head(10)
print(val_subject_counts)

print("\n" + "=" * 80)
print("✓ STAGE C 完成!")
print("=" * 80)
