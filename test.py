import pickle

with open('data/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

print(f"训练样本数: {len(train_data)}")
print(f"样本列: {list(train_data.columns)}")

# 查看第一个样本
sample = train_data.iloc[0]
print(f"HR 序列长度: {len(sample['HR_normalized'])}")