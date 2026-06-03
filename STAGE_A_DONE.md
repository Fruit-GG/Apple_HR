# 数据处理阶段 A - 使用指南

## 处理完成情况

✅ **阶段 A 已完成**

### 处理统计

| 指标 | 值 |
|------|-----|
| 输入 workouts | 992 |
| 处理成功 | 992 (100%) |
| 输出文件格式 | pickle + CSV |
| 平均序列长度 | 109.8 (中位数 110) |
| 时间分辨率 | 10 秒 |

### 输出文件

1. **`data/processed_data.pkl`** (6.7 MB)
   - 完整的结构化数据，包含所有列表列
   - 包含字段：ID_test, ID, Age, Sex, Weight, Height, Humidity, Temperature
   - 时间序列字段：time_grid, Speed, HR, VO2, VCO2, RR, VE

2. **`data/processed_metadata.csv`** (51 KB)
   - 元数据和统计信息，易于快速查看
   - 包含：ID_test, ID, duration, n_measurements, n_interpolated_points

3. **`data/data_quality_report.txt`**
   - 详细的数据质量检查报告

---

## 如何使用处理后的数据

### 方式 1：Python 中加载

```python
import pickle
import pandas as pd

# 加载完整数据
with open('data/processed_data.pkl', 'rb') as f:
    df = pickle.load(f)

# 查看基本信息
print(df.shape)
print(df.columns)

# 访问单个 workout
workout = df.iloc[0]
print(f"ID_test: {workout['ID_test']}")
print(f"HR 序列: {workout['HR']}")
print(f"Speed 序列: {workout['Speed']}")
print(f"Time grid: {workout['time_grid']}")

# 访问个体特征
print(f"Age: {workout['Age']}")
print(f"Sex: {workout['Sex']}")
print(f"Weight: {workout['Weight']}")
print(f"Height: {workout['Height']}")

# 访问环境条件
print(f"Temperature: {workout['Temperature']}")
print(f"Humidity: {workout['Humidity']}")
```

### 方式 2：查看元数据

```python
# 快速加载元数据
df_meta = pd.read_csv('data/processed_metadata.csv')
print(df_meta.head(10))
print(df_meta.describe())
```

---

## 数据结构说明

### 个体信息
- `ID`：个体编号
- `Age`：年龄（最小 10.8，最大 63.0）
- `Sex`：性别（0/1 二值编码）
- `Weight`：体重（kg）
- `Height`：身高（cm）

### Workout 元数据
- `ID_test`：workout 编号
- `duration`：workout 总时长（秒）
- `n_measurements`：原始测量点数
- `n_interpolated_points`：插值后点数（通常 50-172）

### 环境条件
- `Temperature`：温度（℃）
- `Humidity`：湿度（%）

### 时间序列主要变量
- `time_grid`：统一时间网格（秒，10秒间隔）
- `Speed`：运动强度，对应 $I(t)$
- `HR`：心率，对应 $HR(t)$（完整率 100%）

### 时间序列辅助变量
- `VO2`：氧耗量（完整率 98.9%）
- `VCO2`：二氧化碳产生（完整率 98.9%）
- `RR`：呼吸频率（完整率 100%）
- `VE`：分钟通气量（完整率 100%）

---

## 后续阶段 (B/C/D)

模型开发建议顺序：

1. **阶段 B**：数据标准化和特征工程
   - HR 缩放：按论文方式或标准化
   - Speed 归一化
   - 个体特征标准化
   - 生成派生特征（如心率变异性）

2. **阶段 C**：训练/验证/测试划分
   - Subject-level 划分（避免数据泄露）
   - Chronological 划分（按时间序列）
   - 可选：Stratified split（按年龄等维度）

3. **阶段 D**：建立基线模型
   - 线性/树模型：HR ~ {Speed, Temperature, Humidity, Age, Sex}
   - 序列模型：LSTM/GRU for HR 时间序列预测
   - 简单 ODE：不带个体化的 ODE baseline

4. **阶段 E/F**：个体化 ODE 模型 + 结果分析

---

## 常见问题

**Q: 为什么有些 VO2/VCO2 为 None？**
A: 原始数据中这两个字段缺失了约 0.5% 的值（11 个 workouts）。在建模时需要处理这些情况。

**Q: 时间序列长度不一致怎么办？**
A: 序列长度 50-172 是正常的（workout 时长不同）。模型需要支持变长序列，或在批处理时进行 padding/masking。

**Q: 如何验证数据的准确性？**
A: 参考 `data/data_quality_report.txt`，所有 992 个 workouts 都成功处理。可抽样检查几个 workout 的原始数据和插值结果。

---

## 下一步

进行 **阶段 B：数据预处理** 时，会完成：
- 特征标准化
- 异常值过滤
- 派生特征生成

输入你要进行阶段 B 吗？
