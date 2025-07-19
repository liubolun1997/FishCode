import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 加载数据并筛选
df = pd.read_csv('D:/毕设/TON_IoT datasets/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv')
df = df.dropna()

# 仅使用已知攻击和正常数据训练
train_df = df[~df['type'].isin(['normal','xss'])]
X = train_df.select_dtypes(include=[np.number])
y = train_df['type']

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 划分训练集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 训练 XGBoost 模型
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train_scaled, y_train)

# ============ 2. 使用 XSS 类型作为“未知攻击”检测 ============

# 提取 XSS 攻击数据
unknown_df = df[df['type'].isin(['xss'])].sample(n=200, random_state=42)
print(f"验证使用的数据结构：{unknown_df['type'].value_counts()}")
X_unknown = unknown_df.select_dtypes(include=[np.number])

# 如果为空则提示
if X_unknown.empty:
    print(" 未找到任何 xss 类型数据！请检查数据集。")
    exit()

# 使用训练时的 scaler 进行 transform（不要 fit_transform）
X_unknown_scaled = scaler.transform(X_unknown)

# ============ 3. 使用 XGBoost + 置信度判断未知攻击 ============

# 阈值定义（可调）
train_probs = xgb.predict_proba(X_train_scaled)
train_max_proba = np.max(train_probs, axis=1)

# 设置阈值为训练集 max_proba 的某个分位数
confidence_threshold = np.percentile(train_max_proba, 5)  # 下5%置信度作为“可疑”阈值
print(f" 动态设定的阈值为: {confidence_threshold:.4f}")

# 获取每个样本的类别概率
probs = xgb.predict_proba(X_unknown_scaled)

# 最大类别概率（置信度）
max_probs = np.max(probs, axis=1)

# 判定未知攻击（置信度低于阈值）
is_unknown = max_probs < confidence_threshold

# 统计数量
num_unknown = np.sum(is_unknown)
num_known = len(X_unknown_scaled) - num_unknown

# 输出统计结果
print("=== XGBoost 预测结果（xss 作为未知攻击） ===")
print(f" 预测为已知攻击的数量: {num_known}")
print(f" 预测为未知攻击的数量: {num_unknown}")
print(f" 总数: {len(X_unknown_scaled)}")

# 可选：输出比例
print(f"已知攻击占比: {num_known / len(X_unknown_scaled) * 100:.2f}%")
print(f"未知攻击占比: {num_unknown / len(X_unknown_scaled) * 100:.2f}%")
