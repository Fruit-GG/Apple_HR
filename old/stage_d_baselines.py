#!/usr/bin/env python3
"""
Stage D: Baseline Modeling
================================

在阶段 B 的预处理数据上，构建几个简单基线：
1. Global Mean Baseline
2. Polynomial Ridge Regression
3. Histogram Gradient Boosting Regression

说明：
- 为避免数据泄露，这里不使用阶段 C 的 train/val/test 文件。
- 直接基于 preprocessed_data.pkl 重新按 subject ID 做一次干净的 70/15/15 subject-level split。
- 预测目标为 HR_normalized，评估时会还原到 BPM 单位。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import warnings

warnings.filterwarnings("ignore")

ROOT = Path("/home/ljp/Desktop/ml-heart-rate-models")
DATA_DIR = ROOT / "data"
RNG_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_subjects: np.ndarray
    val_subjects: np.ndarray
    test_subjects: np.ndarray


def load_preprocessed_data() -> tuple[pd.DataFrame, dict]:
    with open(DATA_DIR / "preprocessed_data.pkl", "rb") as f:
        df = pickle.load(f)
    with open(DATA_DIR / "preprocessing_stats.pkl", "rb") as f:
        stats = pickle.load(f)
    return df, stats


def clean_workouts(df: pd.DataFrame) -> pd.DataFrame:
    """只保留有完整关键序列的样本。"""
    df = df.copy().reset_index(drop=True)

    def ok_list(x):
        return isinstance(x, list) and len(x) > 5

    mask = (
        df["HR_normalized"].apply(ok_list)
        & df["Speed_normalized"].apply(ok_list)
        & df["time_grid"].apply(ok_list)
    )
    df = df.loc[mask].reset_index(drop=True)
    return df


def make_subject_split(df: pd.DataFrame, seed: int = RNG_SEED) -> SplitData:
    """按 subject ID 进行随机划分，确保同一个 subject 不会跨集合。"""
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(df["ID"].unique()))
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_train = int(round(n_subjects * TRAIN_RATIO))
    n_val = int(round(n_subjects * VAL_RATIO))
    n_train = min(n_train, n_subjects)
    n_val = min(n_val, max(0, n_subjects - n_train))
    n_test = n_subjects - n_train - n_val

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]

    train_df = df[df["ID"].isin(train_subjects)].reset_index(drop=True)
    val_df = df[df["ID"].isin(val_subjects)].reset_index(drop=True)
    test_df = df[df["ID"].isin(test_subjects)].reset_index(drop=True)

    return SplitData(
        train=train_df,
        val=val_df,
        test=test_df,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )


def flatten_workouts(df: pd.DataFrame, hr_mean: float, hr_std: float) -> pd.DataFrame:
    """把 workout 级数据展平成 point-level 样本。"""
    records = []

    feature_cols = [
        "time_norm",
        "time_norm_sq",
        "speed_norm",
        "speed_norm_sq",
        "speed_time_interaction",
        "age_norm",
        "weight_norm",
        "height_norm",
        "sex",
        "temperature",
        "humidity",
    ]

    for _, row in df.iterrows():
        t = np.asarray(row["time_grid"], dtype=float)
        if len(t) < 2:
            continue

        t_end = float(np.max(t))
        if t_end <= 0:
            time_norm = np.zeros_like(t, dtype=float)
        else:
            time_norm = t / t_end

        speed = np.asarray(row["Speed_normalized"], dtype=float)
        y = np.asarray(row["HR_normalized"], dtype=float)

        if len(speed) != len(y) or len(time_norm) != len(y):
            # 长度异常则跳过
            continue

        for i in range(len(y)):
            records.append(
                {
                    "workout_id": row["ID_test"],
                    "subject_id": row["ID"],
                    "time_norm": float(time_norm[i]),
                    "time_norm_sq": float(time_norm[i] ** 2),
                    "speed_norm": float(speed[i]),
                    "speed_norm_sq": float(speed[i] ** 2),
                    "speed_time_interaction": float(speed[i] * time_norm[i]),
                    "age_norm": float(row["Age_norm"]),
                    "weight_norm": float(row["Weight_norm"]),
                    "height_norm": float(row["Height_norm"]),
                    "sex": float(row["Sex"]),
                    "temperature": float(row["Temperature"]),
                    "humidity": float(row["Humidity"]),
                    "target_norm": float(y[i]),
                    "target_bpm": float(y[i] * hr_std + hr_mean),
                }
            )

    flat = pd.DataFrame.from_records(records)
    # 只保留训练所需列
    flat = flat[feature_cols + ["workout_id", "subject_id", "target_norm", "target_bpm"]]
    return flat


def evaluate_predictions(y_true_norm: np.ndarray, y_pred_norm: np.ndarray, hr_mean: float, hr_std: float) -> dict:
    y_true_bpm = y_true_norm * hr_std + hr_mean
    y_pred_bpm = y_pred_norm * hr_std + hr_mean

    return {
        "mae_norm": float(mean_absolute_error(y_true_norm, y_pred_norm)),
        "rmse_norm": float(np.sqrt(mean_squared_error(y_true_norm, y_pred_norm))),
        "mae_bpm": float(mean_absolute_error(y_true_bpm, y_pred_bpm)),
        "rmse_bpm": float(np.sqrt(mean_squared_error(y_true_bpm, y_pred_bpm))),
        "r2_bpm": float(r2_score(y_true_bpm, y_pred_bpm)),
    }


def evaluate_by_workout(df_pred: pd.DataFrame) -> dict:
    grp = df_pred.groupby("workout_id")
    workout_mae = grp.apply(lambda g: mean_absolute_error(g["target_bpm"], g["pred_bpm"]))
    workout_rmse = grp.apply(lambda g: np.sqrt(mean_squared_error(g["target_bpm"], g["pred_bpm"])))
    return {
        "workout_mae_median": float(workout_mae.median()),
        "workout_mae_mean": float(workout_mae.mean()),
        "workout_rmse_median": float(workout_rmse.median()),
        "workout_rmse_mean": float(workout_rmse.mean()),
    }


def make_predictions(model, X: pd.DataFrame, y_true: pd.DataFrame, hr_mean: float, hr_std: float) -> pd.DataFrame:
    pred_norm = model.predict(X)
    out = y_true[["workout_id", "subject_id", "target_norm", "target_bpm"]].copy()
    out["pred_norm"] = pred_norm
    out["pred_bpm"] = pred_norm * hr_std + hr_mean
    out["abs_error_bpm"] = np.abs(out["target_bpm"] - out["pred_bpm"])
    return out


def main():
    print("=" * 80)
    print("STAGE D: 基线模型")
    print("=" * 80)

    df, stats = load_preprocessed_data()
    df = clean_workouts(df)

    hr_mean = float(stats["HR_mean"])
    hr_std = float(stats["HR_std"])

    print(f"\n✓ 加载预处理数据: {len(df)} workouts")
    print(f"✓ HR 标准化参数: mean={hr_mean:.2f}, std={hr_std:.2f}")

    split = make_subject_split(df)
    print("\nSubject-level split:")
    print(f"  训练 subjects: {len(split.train_subjects)}")
    print(f"  验证 subjects: {len(split.val_subjects)}")
    print(f"  测试 subjects: {len(split.test_subjects)}")
    print(f"  训练 workouts: {len(split.train)}")
    print(f"  验证 workouts: {len(split.val)}")
    print(f"  测试 workouts: {len(split.test)}")

    # 保存此阶段的 split 信息，便于复现
    stage_d_split = {
        "train_subjects": split.train_subjects.tolist(),
        "val_subjects": split.val_subjects.tolist(),
        "test_subjects": split.test_subjects.tolist(),
        "seed": RNG_SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
    }
    with open(DATA_DIR / "stage_d_split.pkl", "wb") as f:
        pickle.dump(stage_d_split, f)

    # 展平为 point-level 数据
    train_flat = flatten_workouts(split.train, hr_mean, hr_std)
    val_flat = flatten_workouts(split.val, hr_mean, hr_std)
    test_flat = flatten_workouts(split.test, hr_mean, hr_std)

    print("\nPoint-level sample counts:")
    print(f"  train: {len(train_flat)}")
    print(f"  val:   {len(val_flat)}")
    print(f"  test:  {len(test_flat)}")

    feature_cols = [
        "time_norm",
        "time_norm_sq",
        "speed_norm",
        "speed_norm_sq",
        "speed_time_interaction",
        "age_norm",
        "weight_norm",
        "height_norm",
        "sex",
        "temperature",
        "humidity",
    ]

    X_train = train_flat[feature_cols]
    y_train = train_flat["target_norm"].values
    X_val = val_flat[feature_cols]
    y_val = val_flat["target_norm"].values
    X_test = test_flat[feature_cols]
    y_test = test_flat["target_norm"].values

    models = {}

    # 0. Global mean baseline
    mean_pred_val = np.full_like(y_val, fill_value=y_train.mean(), dtype=float)
    mean_pred_test = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    models["GlobalMean"] = {
        "val": evaluate_predictions(y_val, mean_pred_val, hr_mean, hr_std),
        "test": evaluate_predictions(y_test, mean_pred_test, hr_mean, hr_std),
    }

    # 1. Polynomial Ridge Regression
    ridge_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0, random_state=RNG_SEED)),
        ]
    )
    ridge_model.fit(X_train, y_train)
    ridge_val_pred = ridge_model.predict(X_val)
    ridge_test_pred = ridge_model.predict(X_test)
    models["PolyRidge"] = {
        "val": evaluate_predictions(y_val, ridge_val_pred, hr_mean, hr_std),
        "test": evaluate_predictions(y_test, ridge_test_pred, hr_mean, hr_std),
    }

    # 2. Histogram Gradient Boosting
    hgb_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "hgb",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=250,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=RNG_SEED,
                ),
            ),
        ]
    )
    hgb_model.fit(X_train, y_train)
    hgb_val_pred = hgb_model.predict(X_val)
    hgb_test_pred = hgb_model.predict(X_test)
    models["HistGBDT"] = {
        "val": evaluate_predictions(y_val, hgb_val_pred, hr_mean, hr_std),
        "test": evaluate_predictions(y_test, hgb_test_pred, hr_mean, hr_std),
    }

    # 选出验证集最优模型
    model_name_to_val_mae = {k: v["val"]["mae_bpm"] for k, v in models.items()}
    best_model_name = min(model_name_to_val_mae, key=model_name_to_val_mae.get)
    best_val_mae = model_name_to_val_mae[best_model_name]

    if best_model_name == "GlobalMean":
        best_model = None
        best_val_pred = mean_pred_val
        best_test_pred = mean_pred_test
    elif best_model_name == "PolyRidge":
        best_model = ridge_model
        best_val_pred = ridge_val_pred
        best_test_pred = ridge_test_pred
    else:
        best_model = hgb_model
        best_val_pred = hgb_val_pred
        best_test_pred = hgb_test_pred

    # 生成最佳模型的逐点预测表
    val_pred_df = make_predictions(best_model if best_model is not None else ridge_model, X_val, val_flat, hr_mean, hr_std) if best_model_name != "GlobalMean" else val_flat[["workout_id", "subject_id", "target_norm", "target_bpm"]].copy()
    if best_model_name == "GlobalMean":
        val_pred_df["pred_norm"] = y_train.mean()
        val_pred_df["pred_bpm"] = y_train.mean() * hr_std + hr_mean
        val_pred_df["abs_error_bpm"] = np.abs(val_pred_df["target_bpm"] - val_pred_df["pred_bpm"])

    test_pred_df = make_predictions(best_model if best_model is not None else ridge_model, X_test, test_flat, hr_mean, hr_std) if best_model_name != "GlobalMean" else test_flat[["workout_id", "subject_id", "target_norm", "target_bpm"]].copy()
    if best_model_name == "GlobalMean":
        test_pred_df["pred_norm"] = y_train.mean()
        test_pred_df["pred_bpm"] = y_train.mean() * hr_std + hr_mean
        test_pred_df["abs_error_bpm"] = np.abs(test_pred_df["target_bpm"] - test_pred_df["pred_bpm"])

    val_workout_metrics = evaluate_by_workout(val_pred_df)
    test_workout_metrics = evaluate_by_workout(test_pred_df)

    # 保存预测结果
    val_pred_df.to_csv(DATA_DIR / "stage_d_val_predictions.csv", index=False)
    test_pred_df.to_csv(DATA_DIR / "stage_d_test_predictions.csv", index=False)

    # 保存模型
    if best_model_name == "PolyRidge":
        with open(DATA_DIR / "stage_d_best_model.pkl", "wb") as f:
            pickle.dump(ridge_model, f)
    elif best_model_name == "HistGBDT":
        with open(DATA_DIR / "stage_d_best_model.pkl", "wb") as f:
            pickle.dump(hgb_model, f)

    # 汇总成表
    metrics_rows = []
    for name, res in models.items():
        metrics_rows.append(
            {
                "model": name,
                "val_mae_bpm": res["val"]["mae_bpm"],
                "val_rmse_bpm": res["val"]["rmse_bpm"],
                "val_r2_bpm": res["val"]["r2_bpm"],
                "test_mae_bpm": res["test"]["mae_bpm"],
                "test_rmse_bpm": res["test"]["rmse_bpm"],
                "test_r2_bpm": res["test"]["r2_bpm"],
            }
        )
    metrics_df = pd.DataFrame(metrics_rows).sort_values("val_mae_bpm")
    metrics_df.to_csv(DATA_DIR / "stage_d_metrics.csv", index=False)

    # 生成文本报告
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("STAGE D 基线模型报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("1. 数据划分")
    report_lines.append("-" * 80)
    report_lines.append(f"  训练 subjects: {len(split.train_subjects)}")
    report_lines.append(f"  验证 subjects: {len(split.val_subjects)}")
    report_lines.append(f"  测试 subjects: {len(split.test_subjects)}")
    report_lines.append(f"  训练 workouts: {len(split.train)}")
    report_lines.append(f"  验证 workouts: {len(split.val)}")
    report_lines.append(f"  测试 workouts: {len(split.test)}")
    report_lines.append("")

    report_lines.append("2. Point-level 样本数")
    report_lines.append("-" * 80)
    report_lines.append(f"  训练点数: {len(train_flat)}")
    report_lines.append(f"  验证点数: {len(val_flat)}")
    report_lines.append(f"  测试点数: {len(test_flat)}")
    report_lines.append("")

    report_lines.append("3. 模型比较（验证集 -> 测试集）")
    report_lines.append("-" * 80)
    report_lines.append(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    report_lines.append("")

    report_lines.append("4. 最优模型")
    report_lines.append("-" * 80)
    report_lines.append(f"  Best model by val MAE: {best_model_name}")
    report_lines.append(f"  Validation MAE: {best_val_mae:.3f} BPM")
    report_lines.append("")

    report_lines.append("5. 逐 workout 误差（最优模型）")
    report_lines.append("-" * 80)
    report_lines.append(f"  Validation workout MAE median: {val_workout_metrics['workout_mae_median']:.3f} BPM")
    report_lines.append(f"  Validation workout MAE mean:   {val_workout_metrics['workout_mae_mean']:.3f} BPM")
    report_lines.append(f"  Test workout MAE median:        {test_workout_metrics['workout_mae_median']:.3f} BPM")
    report_lines.append(f"  Test workout MAE mean:          {test_workout_metrics['workout_mae_mean']:.3f} BPM")
    report_lines.append("")

    report_lines.append("6. 说明")
    report_lines.append("-" * 80)
    report_lines.append("  - 本阶段使用了 point-level 监督：每个时间点预测对应 HR_normalized。")
    report_lines.append("  - 为避免泄露，重新按 subject ID 划分了训练/验证/测试。")
    report_lines.append("  - 这是 baseline，后续阶段可在此基础上加入 ODE / encoder / 个体表示 z。")
    report_lines.append("")

    report_text = "\n".join(report_lines)
    with open(DATA_DIR / "stage_d_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # 终端输出
    print("\n" + "=" * 80)
    print(report_text)
    print("=" * 80)

    print("\n已保存输出:")
    print(f"  - {DATA_DIR / 'stage_d_split.pkl'}")
    print(f"  - {DATA_DIR / 'stage_d_metrics.csv'}")
    print(f"  - {DATA_DIR / 'stage_d_val_predictions.csv'}")
    print(f"  - {DATA_DIR / 'stage_d_test_predictions.csv'}")
    print(f"  - {DATA_DIR / 'stage_d_report.txt'}")
    if best_model_name in {"PolyRidge", "HistGBDT"}:
        print(f"  - {DATA_DIR / 'stage_d_best_model.pkl'}")

    print("\n✓ STAGE D 完成!")


if __name__ == "__main__":
    main()
