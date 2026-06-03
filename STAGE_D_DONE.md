# 数据处理阶段 D - 基线模型完成

## 处理完成情况

✅ **阶段 D 已完成**

### 📊 模型结果概览

本阶段在阶段 B 的预处理数据上训练了 3 个基线：

1. **Global Mean Baseline**
2. **Polynomial Ridge Regression**
3. **Histogram Gradient Boosting Regression**

### 🏆 最优模型

- **模型**：HistGBDT
- **验证集 MAE**：10.557 BPM
- **测试集 MAE**：11.624 BPM
- **测试集 RMSE**：14.791 BPM
- **测试集 R²**：0.804

### 📁 输出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `data/stage_d_metrics.csv` | 455 B | 模型指标汇总 |
| `data/stage_d_report.txt` | 1.9 KB | 详细报告 |
| `data/stage_d_val_predictions.csv` | 1.2 MB | 验证集逐点预测结果 |
| `data/stage_d_test_predictions.csv` | 1.2 MB | 测试集逐点预测结果 |
| `data/stage_d_best_model.pkl` | 889 KB | 最优基线模型 |
| `data/stage_d_split.pkl` | 1.8 KB | 阶段 D 的 subject 划分信息 |

---

## 模型比较

| 模型 | Val MAE (BPM) | Val RMSE (BPM) | Val R² | Test MAE (BPM) | Test RMSE (BPM) | Test R² |
|------|---------------:|---------------:|-------:|---------------:|----------------:|--------:|
| HistGBDT | 10.557 | 13.509 | 0.827 | 11.624 | 14.791 | 0.804 |
| PolyRidge | 10.598 | 13.402 | 0.830 | 12.079 | 15.322 | 0.790 |
| GlobalMean | 27.751 | 32.484 | ~0.000 | 28.612 | 33.452 | ~0.000 |

---

## Stage D 说明

### 输入

- `data/preprocessed_data.pkl`
- `data/preprocessing_stats.pkl`

### 处理方式

1. 读取阶段 B 输出。
2. 重新按 subject ID 做一次干净的 70/15/15 划分。
3. 将每个 workout 展平成 point-level 样本。
4. 以 `HR_normalized` 为目标，训练 baseline 模型。
5. 在评估时还原到 BPM 单位。

### 特征

用于建模的 point-level 特征包括：

- `time_norm`
- `time_norm_sq`
- `speed_norm`
- `speed_norm_sq`
- `speed_time_interaction`
- `age_norm`
- `weight_norm`
- `height_norm`
- `sex`
- `temperature`
- `humidity`

---

## 结论

- 最简单的全局均值基线效果较差。
- 加入时间、速度和个体特征后，树模型明显更好。
- 当前最优 baseline 为 **HistGBDT**，测试集 MAE 约 **11.6 BPM**。
- 这可以作为后续 Stage E 个体化 ODE 模型的比较基线。

---

## 下一步建议

1. **阶段 E**：开始构建个体化 ODE 模型。
2. **对比分析**：将 ODE 模型和本阶段基线在 MAE / RMSE 上比较。
3. **可解释性分析**：检查速度、温度、湿度和个体特征对预测的影响。

