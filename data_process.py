from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import pickle
import warnings
import matplotlib.pyplot as plt
from ode.data import WorkoutDataset, WorkoutDatasetConfig, make_dataloaders
from ode.ode import ODEModel, OdeConfig
from ode.trainer import train_ode_model

def load_data():
    subject_info = pd.read_csv('data/subject-info.csv')
    test_measure = pd.read_csv('data/test_measure.csv')

    print(f"\n✓ subject-info.csv: {subject_info.shape[0]} 行")
    print(f"✓ test_measure.csv: {test_measure.shape[0]} 行")
    print(f"\n 列名 (subject-info):\n  {list(subject_info.columns)}")
    print(f"\n 列名 (test_measure):\n  {list(test_measure.columns)}")

    n_subjects = test_measure['ID'].nunique()
    n_workouts = test_measure['ID_test'].nunique()

    print(f"\n✓ 独特 subject (ID) 参与者数量: {n_subjects}")
    print(f"✓ 独特 workouts (ID_test) 测试数量: {n_workouts}")

    return subject_info, test_measure

def process_workout_data(subject_info, test_measure):
    # 数据分组
    
    # 主题特征查找表 ，ID_test为键，{}中为对应的值 进行构建
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
        # Map 函数 对每个 ID_test 进行映射，获取对应的特征值，如果没有则返回 NaN
        test_measure_with_info[col] = test_measure_with_info['ID_test'].map(
            lambda x: subject_feature_lookup.get(x, {}).get(col, np.nan)
        )
    
    return test_measure_with_info

def data_clean(test_measure_with_info):
    """
    清理缺失的数据
    清理 心率 < 0.3s 和 > 2s的异常数据
    
    """
    current_workout = len(test_measure_with_info['ID_test'].unique())
    print(f"\n✓ 当前 workout 数量: {current_workout}")
    
    # 删除包含任何缺失值的行
    workouts_with_na = test_measure_with_info[test_measure_with_info.isna().any(axis=1)]['ID_test'].unique()
    test_measure_cleaned = test_measure_with_info[
        ~test_measure_with_info['ID_test'].isin(workouts_with_na)
    ]
    current_workout_after_dropna = len(test_measure_cleaned['ID_test'].unique())
    print(f"✓ 删除缺失值后 workout 数量: {current_workout_after_dropna}")

    # 过滤掉心率 < 0.3s 或 > 2s 的异常数据
    abnormal_workouts = test_measure_cleaned[
        (test_measure_cleaned['HR'] < 30) | (test_measure_cleaned['HR'] > 200)
    ]['ID_test'].unique()
    
    test_measure_cleaned = test_measure_cleaned[
        ~test_measure_cleaned['ID_test'].isin(abnormal_workouts)
    ]

    current_workout_after_filtering = len(test_measure_cleaned['ID_test'].unique())
    print(f"✓ 过滤异常心率数据后 workout 数量: {current_workout_after_filtering}")

    print(f"\n✓ 数据清理完成: {test_measure_cleaned.shape[0]} 行剩余")
    return test_measure_cleaned
    
def process_workout(group):
    """
    对单个workout进行处理,输入的是一条workout的数据，包含多个时间点的心率数据和对应的特征信息
    1、按照时间排序 15min左右的跑步机数据， 应该在900+行
    2、插值到固定的时间点
    3、特殊处理：归一化心率、新建时间戳
    4、返回处理后的数据
    """
    # 检查数据量
    if len(group) < 10:
        warnings.warn(f"Workout {group['ID_test'].iloc[0]} 数据点过少: {len(group)} 行")
        return None
    
    # 先按时间排好顺序，再把行号重新变成 0,1,2,...，并丢掉旧的乱序 index
    group = group.sort_values('time').reset_index(drop=True)
    
    # 插值到固定的时间点，假设我们要插值到 1 秒
    time_vals = group['time'].values.astype(float)
    
    t_start = time_vals[0]
    t_end = time_vals[-1]
    duration = t_end - t_start
    
    # 生成时间网格，假设我们要插值到 1 秒的时间点
    target_grid = np.arange(t_start, t_end + 1, 1)
    
    # 重排数据结构, 对应 Dataloader 需要的输入格式
    
    result = {}
    # 基础信息
    result["subject_id"] = group['ID'].iloc[0]  # 用户ID
    result['workout_id'] = group['ID_test'].iloc[0] # workout ID
    
    # 插值信息
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
    
    # 时间序列的信息
    result['time_grid'] = target_grid.tolist()
    for col in ['Speed', 'HR', 'VO2', 'VCO2', 'RR', 'VE']:
        if col not in group.columns:
            warnings.warn(f"Workout {group['ID_test'].iloc[0]} 缺少列 {col}")
            continue
        
        try:
            y_vals = group[col].values.astype(float)

            f_interp = interp1d(
                time_vals,
                y_vals,
                kind='linear',
                bounds_error=True
            )

            result[col] = f_interp(target_grid).tolist()

        except Exception as e:
            raise RuntimeError(
                f"ID_test={result['ID_test']} 在列 {col} 插值失败"
            ) from e
    
    # 心率归一化，统计分析显示，平均心率在 138 左右，标准差在 32 左右
    result["heart_rate_normalized"] = ((np.array(result['HR']) - 138) / 32).tolist()  
        
    return result

def group_workout_data(test_measure_cleaned):
    # 按照 ID_test 分组
    grouped = test_measure_cleaned.groupby('ID_test')
    
    # 创建一个新的 DataFrame 来存储每个 workout 的特征
    workout_features = []
    
    # 用于生成顺序时间戳的计数器
    base_time = datetime(2024, 1, 1, 0, 0, 0)   # 基准起始时间
    workout_counter = 0
    
    
    # 处理所有 workouts
    skipped_count = 0
    for idx, (wid, group) in enumerate(grouped):
        if (idx + 1) % 100 == 0:
            print(f"  已处理: {idx + 1}/{grouped.ngroups}")
        
        # 对每个 workout 进行处理
        processed = process_workout(group)
        if processed is not None:
            # 简单为该 workout 增加一个模拟的开始时间（按处理顺序递增一天），适配DataLoader
            processed['time_of_start_column'] = base_time + timedelta(days=workout_counter)
            workout_counter += 1
            workout_features.append(processed)
        else:
            skipped_count += 1
    
    print(f"✓ 成功处理: {len(workout_features)} 个 workouts")
    print(f"✗ 跳过: {skipped_count} 个 workouts (数据质量问题)")

    return pd.DataFrame(workout_features)

def analyze_workout_features(workout_df):
    """
    对插值后的 workout 数据进行统计分析
    
    输入:
        workout_df: pd.DataFrame，每行是一条 workout（process_workout 输出）
    输出:
        stats: dict，包含每个序列变量的统计信息
    """
    sequence_cols = ['Speed', 'HR', 'VO2', 'VCO2', 'RR', 'VE']
    stats = {}

    for col in sequence_cols:
        if col not in workout_df.columns:
            continue
        
        # 拼接所有 workout 的序列，方便统计
        all_values = np.concatenate(workout_df[col].dropna().values)
        
        stats[col] = {
            'count': len(all_values),
            'mean': np.mean(all_values),
            'std': np.std(all_values),
            'min': np.min(all_values),
            '25%': np.percentile(all_values, 25),
            '50%': np.median(all_values),
            '75%': np.percentile(all_values, 75),
            'max': np.max(all_values)
        }

    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv("workout_stats.csv", index=True)
    return stats_df
 
def plot_interpolation_comparison(workout_df, test_id, raw_data):
    """
    绘制插值前后的对比图
    
    输入:
        workout_df: pd.DataFrame，process_workout 输出的结果
        test_id: str，想要查看的 workout ID_test
        raw_data: 原始 DataFrame，包含对应的时间和序列列
    """
    # 找到处理后的 workout
    workout = workout_df[workout_df['workout_id'] == test_id]
    if workout.empty:
        raise ValueError(f"未找到 ID_test={test_id} 的处理数据")
    
    workout = workout.iloc[0]
    
    time_grid = np.array(workout['time_grid'])
    
    sequence_cols = ['Speed', 'HR', 'VO2', 'VCO2', 'RR', 'VE']
    
    fig, axes = plt.subplots(len(sequence_cols), 1, figsize=(12, 3*len(sequence_cols)))
    
    if len(sequence_cols) == 1:
        axes = [axes]
    
    # 原始数据
    raw_group = raw_data[raw_data['ID_test'] == test_id].sort_values('time')
    
    for ax, col in zip(axes, sequence_cols):
        if col not in workout:
            continue
        
        # 插值后的序列
        interp_vals = workout[col]
        
        # 原始序列
        if col in raw_group.columns:
            raw_time = raw_group['time'].values.astype(float)
            raw_vals = raw_group[col].values.astype(float)
            ax.plot(raw_time, raw_vals, 'o-', label='origin', alpha=0.5)
        
        ax.plot(time_grid, interp_vals, '-', label='interp', linewidth=2)
        ax.set_title(f"{col} Comparison of interp - ID_test={test_id}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"interpolation_comparison_{test_id}.png")
    plt.show()

def load_with_dataloader(workout_features):
    
    df = workout_features.rename(columns={
        'time_of_start_column': 'time_start',
        'HR': 'heart_rate'
    })

    config = WorkoutDatasetConfig()
    config.activity_columns = ["Speed"]   
    config.weather_columns = ["Humidity", "Temperature"]
    config.history_max_length = 512
    
    total = len(df)
    train_len = int(total * 0.8)
    train_dataset = WorkoutDataset(df.iloc[:train_len], config)
    test_dataset = WorkoutDataset(df.iloc[train_len:], config)


    
    return train_dataset, test_dataset, config, df

def main():
    subject_info, test_measure = load_data()
    test_measure_with_info = process_workout_data(subject_info, test_measure)
    test_measure_cleaned = data_clean(test_measure_with_info)
    workout_features = group_workout_data(test_measure_cleaned)
    
    stats_df = analyze_workout_features(workout_features)
    # plot_interpolation_comparison(workout_features, test_id='484_1', raw_data=test_measure_cleaned)
    
    train_dataset, test_dataset, data_config_train, df_tmp = load_with_dataloader(workout_features)
    
    train_dataloader, test_dataloader = make_dataloaders(train_dataset, test_dataset, batch_size=8)
    
    ode_config = OdeConfig(
        data_config_train,
        learning_rate=1e-3,
        seed=0,
        n_epochs=10,
        encoder_embedding_dim=8,
        subject_embedding_dim=4,
    )

    model = ODEModel(
            workouts_info=df_tmp[["subject_id", "workout_id"]],
            config=ode_config,)
    
    total = len(df_tmp)
    train_len = int(total * 0.8)
    train_workout_ids = set(df_tmp.iloc[:train_len]["workout_id"].values)
    res = train_ode_model(model, train_dataloader, test_dataloader, train_workout_ids)

if __name__ == "__main__":
    main()