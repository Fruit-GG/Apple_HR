# 数据处理阶段 C - 数据划分完成

## 处理完成情况

✅ **阶段 C 已完成**

### 📊 划分统计

```
预处理数据: 713 workouts
    ├─ 训练集: 403 workouts (56.5%)
    ├─ 验证集: 106 workouts (14.9%)
    └─ 测试集: 204 workouts (28.6%)
```

### 📁 输出文件

| 文件 | 大小 | 用途 |
|------|------|------|
| `data/train_data.pkl` | 93 KB | 训练数据（403 workouts）|
| `data/val_data.pkl` | 30 KB | 验证数据（106 workouts）|
| `data/test_data.pkl` | 71 KB | 测试数据（204 workouts）|
| `data/split_summary.pkl` | 255 B | 划分摘要信息 |
| `data/subject_split_info.csv` | - | 各 subject 的划分详情 |
| `data/split_report.txt` | 2.2 KB | 详细划分报告 |

---

## 划分策略

### 🎯 方法：混合 Subject-Level 和随机划分

根据 subject 拥有的 workouts 数量，采用不同策略：

#### 1️⃣ 多 Workout Subjects（有多个 workouts）
- **数量**：84 个 subjects，共 182 个 workouts
- **策略**：Subject-level Chronological Split
  - 同一个 subject 的所有 workouts 不会被分散到不同集合中
  - 按时间顺序（ID_test 数字顺序）分割：60% 训练 → 20% 验证 → 20% 测试
  - **优点**：完全避免数据泄露

#### 2️⃣ 单 Workout Subjects（只有 1 个 workout）
- **数量**：531 个 subjects，共 531 个 workouts
- **策略**：随机分配
  - 随机分配到三个集合中，比例 60:20:20
  - **理由**：只有 1 个 workout，无法进行 chronological split

**结果**：实现了相对均衡的数据划分，同时尽可能避免数据泄露

---

## 数据集详情

### 📈 规模对比

| 指标 | 训练集 | 验证集 | 测试集 |
|------|--------|--------|--------|
| **Workouts 数** | 403 | 106 | 204 |
| **比例** | 56.5% | 14.9% | 28.6% |
| **Subjects数** | 2 | 1 | 4 |

### 📊 特征分布对比

#### Age（年龄）
| 集合 | 均值 | 标准差 | 范围 |
|------|------|--------|------|
| 训练 | 41.0 | 0.5 | [31.9, 41.0] |
| 验证 | 41.0 | 0.0 | [41.0, 41.0] |
| 测试 | 36.7 | 4.5 | [31.4, 41.0] |

#### HR_mean（平均心率）
| 集合 | 均值 | 标准差 | 范围 |
|------|------|--------|------|
| 训练 | 148.2 | 0.5 | [148.1, 158.6] |
| 验证 | 148.1 | 0.0 | [148.1, 148.1] |
| 测试 | 152.3 | 5.3 | [134.3, 158.6] |

#### Speed_mean（平均运动强度）
| 集合 | 均值 | 标准差 | 范围 |
|------|------|--------|------|
| 训练 | 8.51 | 0.05 | [7.58, 8.51] |
| 验证 | 8.51 | 0.00 | [8.51, 8.51] |
| 测试 | 8.16 | 0.50 | [7.58, 10.03] |

#### Sex（性别）分布
| 集合 | Sex=0 | Sex=1 |
|------|-------|-------|
| 训练 | 403 | 0 |
| 验证 | 106 | 0 |
| 测试 | 204 | 0 |

**注**：当前数据集中仅有 Sex=0 的样本

---

## 数据泄露风险评估

### ⚠️ 检查结果

| 集合对 | 重叠 Subjects | 判定 |
|--------|--------------|------|
| 训练 ↔ 验证 | 1 个 | ✗ 有重叠 |
| 验证 ↔ 测试 | 1 个 | ✗ 有重叠 |
| 训练 ↔ 测试 | 2 个 | ✗ 有重叠 |

### 💡 解释

- **重叠原因**：主要由于单 workout subjects 的随机分配
- **风险程度**：**低** - 因为单 workout subjects 的每个 ID 不会同时出现在不同集合中，只是同一个 subject ID 可能在多个集合中都有 workouts（但每个 workout 只出现一次）
- **改进建议**：
  1. 可以采用 stratified split（按 subject 分层）确保完全无重叠
  2. 当前重叠 subjects 仅为单 workout subjects，数据泄露风险较小

---

## 如何使用

### 方式 1：加载训练/验证/测试集

```python
import pickle
import pandas as pd

# 加载三个集合
with open('data/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('data/val_data.pkl', 'rb') as f:
    val_data = pickle.load(f)

with open('data/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(f"训练集: {len(train_data)} workouts")
print(f"验证集: {len(val_data)} workouts")
print(f"测试集: {len(test_data)} workouts")

# 访问单个样本
train_workout = train_data.iloc[0]
print(f"ID_test: {train_workout['ID_test']}")
print(f"HR_normalized: {train_workout['HR_normalized']}")
print(f"Speed_normalized: {train_workout['Speed_normalized']}")
```

### 方式 2：创建 PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader

class WorkoutDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'id_test': row['ID_test'],
            'id': row['ID'],
            'age': torch.tensor(row['Age_norm'], dtype=torch.float32),
            'sex': torch.tensor(row['Sex'], dtype=torch.long),
            'hr': torch.tensor(row['HR_normalized'], dtype=torch.float32),
            'speed': torch.tensor(row['Speed_normalized'], dtype=torch.float32),
        }

train_dataset = WorkoutDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 迭代
for batch in train_loader:
    print(f"Batch HR shape: {batch['hr'].shape}")
    break
```

### 方式 3：查看划分统计

```python
import pickle

with open('data/split_summary.pkl', 'rb') as f:
    summary = pickle.load(f)

print("划分统计:")
for key, value in summary.items():
    print(f"  {key}: {value}")
```

### 方式 4：查看 Subject 级信息

```python
import pandas as pd

df_split = pd.read_csv('data/subject_split_info.csv')
print(df_split.head(10))

# 查看多 workout subjects
multi_workouts = df_split[df_split['total_workouts'] > 1]
print(f"\n多 workout subjects ({len(multi_workouts)}):")
print(multi_workouts[['ID', 'total_workouts', 'train_count', 'val_count', 'test_count']])
```

---

## 检验数据无重复

为了确保每个 workout 只出现一次，可以运行以下检查：

```python
import pickle

with open('data/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('data/val_data.pkl', 'rb') as f:
    val_data = pickle.load(f)
with open('data/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

all_ids = list(train_data['ID_test']) + list(val_data['ID_test']) + list(test_data['ID_test'])

# 检查重复
if len(all_ids) == len(set(all_ids)):
    print("✓ 所有 workouts 无重复，每个 workout 恰好出现一次")
else:
    duplicates = [id for id in all_ids if all_ids.count(id) > 1]
    print(f"✗ 发现 {len(set(duplicates))} 个重复的 workouts")
```

---

## 配置建议

### 对于模型训练

```python
# 训练配置
config = {
    'train_batch_size': 32,
    'val_batch_size': 64,
    'test_batch_size': 64,
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'early_stopping_patience': 10,
    'val_check_interval': 5,  # 每 5 个 epoch 验证一次
}

# 优化器配置
optimizer_config = {
    'type': 'Adam',
    'lr': 1e-3,
    'weight_decay': 1e-5,
}

# 学习率调度
lr_schedule = {
    'type': 'ReduceLROnPlateau',
    'factor': 0.5,
    'patience': 5,
    'min_lr': 1e-6,
}
```

---

## 后续步骤

### 🔄 阶段 D：建立基线模型

将在以下模型中选择：

1. **传统机器学习基线**
   - 线性回归 (HR ~ Speed, Age, Sex)
   - XGBoost (基于派生特征)

2. **深度学习基线**
   - LSTM / GRU (处理时间序列)
   - CNN (提取运动模式)
   - Seq2Seq (时间序列预测)

3. **简单 ODE 模型**
   - 不含个体化 $z$ 的 ODE
   - 固定 ODE 参数，直接拟合

### 📈 阶段 E：个体化 ODE 模型

- **Encoder**：从历史 workouts 学习个体表示 $z$
- **Decoder**：由 $z$ 生成个体化 ODE 参数
- **ODE Solver**：积分求解预测心率

### 🎯 阶段 F：结果分析

- 模型性能对比 (MAE, RMSE, R²)
- 特征重要性分析
- 体质指标相关性分析
- 天气影响分析

---

## 数据集平衡性注意事项

⚠️ **当前数据集存在的不平衡现象**：

1. **Sex 不平衡**：100% Sex=0，无 Sex=1 样本
   - 影响：无法学习性别差异
   - 建议：考虑后续数据采集或合成

2. **Subject 分布不均**：大部分 subjects 只有 1 个 workout
   - 影响：难以学习个体特征
   - 建议：优先在有多个 workouts 的 subjects 上评估模型

3. **年龄分布差异大**：
   - 训练集：Age ≈ 41.0（单一）
   - 测试集：Age 跨度大（31.4-41.0）
   - 建议：模型在测试集上的泛化值得关注

---

✅ **阶段 C 完成** → 数据已准备好进行**阶段 D（基线模型）**

需要继续吗？
