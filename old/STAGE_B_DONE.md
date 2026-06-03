# 数据处理阶段 B - 预处理完成

## 处理完成情况

✅ **阶段 B 已完成**

### 📊 处理统计

| 阶段 | 输入 | 输出 | 过滤率 |
|------|------|------|--------|
| **初始数据** | 992 workouts | - | - |
| **功能性数据** | 992 | 889 | HR范围异常：10.4% |
| **最终数据** | 889 | 713 | Speed异常：19.8% |

### 📁 输出文件

1. **`data/preprocessed_data.pkl`** (6.2 MB)
   - 完整的标准化数据，包含所有时间序列
   - 包含原始特征 + 标准化特征 + 派生特征

2. **`data/preprocessing_stats.pkl`** (1.3 KB)
   - 所有标准化参数的字典
   - 用于对测试集应用相同的转换

3. **`data/preprocessed_metadata.csv`** (86 KB)
   - 元数据 + 派生特征，易于查看

4. **`data/preprocessing_report.txt`**
   - 详细的预处理报告

---

## 数据标准化方案

### 1️⃣ 个体特征（连续变量）

使用 **Z-score 标准化**：$x_{norm} = \frac{x - \mu}{\sigma}$

| 特征 | 均值 | 标准差 |
|------|------|--------|
| Age | 27.92 | 9.85 |
| Weight | 73.78 | 11.83 |
| Height | 175.36 | 7.93 |

**新列**：`Age_norm`, `Weight_norm`, `Height_norm`

### 2️⃣ 心率（时间序列）

使用 **Z-score 标准化**：$HR_{norm}(t) = \frac{HR(t) - \mu_{HR}}{\sigma_{HR}}$

| 统计量 | 数值 |
|--------|------|
| 均值 | 140.21 BPM |
| 标准差 | 32.66 BPM |
| 范围 | [30.00, 218.00] BPM |

**新列**：`HR_normalized` (包含 100+ 个时间点的归一化序列)

### 3️⃣ 运动强度 Speed（时间序列）

使用 **Min-Max 归一化**：$Speed_{norm}(t) = \frac{Speed(t) - min}{max - min}$

| 统计量 | 数值 |
|--------|------|
| 均值 | 8.80 |
| 标准差 | 4.14 |
| 范围 | [2.40, 23.00] |

**新列**：`Speed_normalized` (范围 [0, 1])

### 4️⃣ 辅助生理变量

使用 **Z-score 标准化**

| 变量 | 均值 | 标准差 | 非空率 |
|------|------|--------|--------|
| VO2 | 2132.25 | 954.47 | 98.0% |
| VCO2 | 2114.83 | 1099.21 | 98.0% |
| RR | 32.72 | 11.52 | 100% |
| VE | 60.64 | 31.41 | 100% |

**新列**：`VO2_normalized`, `VCO2_normalized`, `RR_normalized`, `VE_normalized`

### 5️⃣ 性别（分类变量）

已进行 **二值编码**，范围 {0, 1}

- Sex=0: 611 个样本
- Sex=1: 102 个样本

---

## 派生特征

新增的统计特征，用于增强模型的表示能力：

| 特征 | 定义 | 均值 | 范围 |
|------|------|------|------|
| **HR_var** | 心率标准差（变异性） | 29.44 | [13.30, 44.11] |
| **HR_mean** | 平均心率 | 140.97 | [102.46, 173.74] |
| **Speed_mean** | 平均运动强度 | 8.71 | [4.95, 11.85] |
| **HR_slope** | 心率线性趋势 | 0.66 | [-0.09, 1.32] |

每个派生特征都已标准化（`*_norm` 版本）。

---

## 数据质量过滤

应用了 5 层过滤：

1. **HR 缺失**：0 个过滤（100% 保留）
2. **Speed 缺失**：0 个过滤（100% 保留）
3. **序列长度 < 10 点**：0 个过滤（100% 保留）
4. **HR 范围异常**：103 个过滤（10.4%）
   - 正常范围：[30, 220] BPM
5. **Speed 范围异常**：176 个过滤（19.8%）
   - 正常范围：(0, 30] km/h

👉 **最终数据质量很高**：713/992 (71.8%) 的有效 workouts

---

## 如何使用预处理数据

### 方式 1：Python 中加载

```python
import pickle
import pandas as pd
import numpy as np

# 加载预处理数据
with open('data/preprocessed_data.pkl', 'rb') as f:
    df = pickle.load(f)

# 加载统计参数
with open('data/preprocessing_stats.pkl', 'rb') as f:
    stats = pickle.load(f)

# 访问第一个 workout
workout = df.iloc[0]

# ✓ 原始个体特征
print(f"Age: {workout['Age']}")
print(f"Sex: {workout['Sex']}")

# ✓ 标准化个体特征
print(f"Age_norm: {workout['Age_norm']:.3f}")

# ✓ 原始时间序列
print(f"HR: {workout['HR']}")  # [7.8, 12.5, ...]
print(f"Speed: {workout['Speed']}")  # [5.0, 5.1, ...]

# ✓ 标准化时间序列
print(f"HR_normalized: {workout['HR_normalized']}")  # [-0.986, -0.915, ...]
print(f"Speed_normalized: {workout['Speed_normalized']}")  # [0.126, 0.126, ...]

# ✓ 派生特征
print(f"HR_mean: {workout['HR_mean']:.2f}")
print(f"HR_var: {workout['HR_var']:.2f}")
print(f"Speed_mean: {workout['Speed_mean']:.2f}")
print(f"HR_slope: {workout['HR_slope']:.3f}")
```

### 方式 2：对新数据应用相同的标准化

```python
# 对测试数据应用相同的标准化
new_age = 35
new_age_norm = (new_age - stats['Age_mean']) / (stats['Age_std'] + 1e-8)

new_hr_values = [120, 125, 130, 135, 140]
new_hr_norm = [(hr - stats['HR_mean']) / (stats['HR_std'] + 1e-8) 
               for hr in new_hr_values]

new_speed_values = [8.0, 8.5, 9.0]
new_speed_norm = [(s - stats['Speed_min']) / (stats['Speed_max'] - stats['Speed_min']) 
                  for s in new_speed_values]
```

### 方式 3：快速查看元数据

```python
df_meta = pd.read_csv('data/preprocessed_metadata.csv')
print(df_meta.head(10))
print(df_meta.describe())
```

---

## 特征总览

预处理后的数据共有 **35 列**：

### 原始列 (22 列)
- ID、个体特征：`ID_test`, `ID`, `Age`, `Sex`, `Weight`, `Height`, `Humidity`, `Temperature`
- 时间序列：`time_grid`, `Speed`, `HR`, `VO2`, `VCO2`, `RR`, `VE`
- 元数据：`duration`, `n_measurements`, `n_interpolated_points`
- 派生特征：`HR_var`, `HR_mean`, `Speed_mean`, `HR_slope`

### 标准化列 (13 列)
- 个体特征：`Age_norm`, `Weight_norm`, `Height_norm`
- 时间序列：`HR_normalized`, `Speed_normalized`, `VO2_normalized`, `VCO2_normalized`, `RR_normalized`, `VE_normalized`
- 派生特征：`HR_var_norm`, `HR_mean_norm`, `Speed_mean_norm`, `HR_slope_norm`

---

## 数据特点

### ✅ 优点
- ✓ 100% 的核心变量（HR、Speed、RR、VE）完整
- ✓ 良好的个体多样性（年龄 18-52 岁，体重 41-135 kg）
- ✓ 所有派生特征都有物理意义
- ✓ 标准化参数已保存，便于测试集应用

### ⚠️ 注意事项
- 约 2% 的 VO2/VCO2 缺失（可选特征）
- 数据过滤后从 992 → 713 (28% 过滤)
- 性别不平衡：0 (86%) vs 1 (14%)

---

## 后续步骤

### 🔄 阶段 C：数据划分
将在以下方式中选择：
- **Approach 1**：Subject-level split (避免数据泄露)
- **Approach 2**：Chronological split (按时间顺序)
- **Approach 3**：Stratified split (按年龄等特征)

### 📈 阶段 D：建立基线模型
- 线性回归 / XGBoost 基线
- LSTM 序列模型
- 简单 ODE 模型（不带个体化）

### 🎯 阶段 E/F：重点工作
- 完整的个体化 ODE 模型
- Encoder 学习潜在表示 $z$
- Decoder 生成个体化 ODE 参数

---

## 验证方法

检查数据是否成功标准化：

```bash
# 查看文件大小
ls -lh data/preprocessed*

# 打印统计参数
python3 << EOF
import pickle
with open('data/preprocessing_stats.pkl', 'rb') as f:
    stats = pickle.load(f)
print("标准化参数示例:")
for key in ['Age_mean', 'HR_mean', 'HR_std', 'Speed_min', 'Speed_max']:
    print(f"  {key}: {stats.get(key)}")
EOF
```

---

✅ **阶段 B 完成** → 数据已准备好进行**阶段 C（数据划分）**或**阶段 D（基线模型）**

需要继续吗？
