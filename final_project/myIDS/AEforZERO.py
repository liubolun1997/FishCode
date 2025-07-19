import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ========== 1. 加载数据 ==========
df = pd.read_csv('D:/毕设/TON_IoT datasets/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv')
df = df.dropna()
df = df[df['type'].isin(['dos', 'backdoor', 'ddos','password','scanning'])]

print("标签分布：")
print(df['type'].value_counts())

# ========== 2. 已知攻击类型 ==========
attack_types = ['dos','backdoor','password','scanning']
autoencoders = {}
thresholds = {}
scalers = {}

# ========== 3. 构建 Autoencoder ==========
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu',
                    activity_regularizer=regularizers.l1(1e-5))(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# ========== 4. 训练每类攻击的 AE ==========
for attack in attack_types:
    print(f"\n=== 正在训练 Autoencoder 模型：{attack} ===")
    data = df[df['type'] == attack]
    X = data.select_dtypes(include=[np.number])

    X_train, X_val = train_test_split(X, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    ae = build_autoencoder(input_dim=X.shape[1])
    ae.fit(X_train_scaled, X_train_scaled,
           epochs=50,
           batch_size=256,
           shuffle=True,
           validation_data=(X_val_scaled, X_val_scaled),
           verbose=0)

    X_val_pred = ae.predict(X_val_scaled, verbose=0)
    mse_val = np.mean(np.square(X_val_scaled - X_val_pred), axis=1)
    threshold = np.percentile(mse_val, 95)

    autoencoders[attack] = ae
    thresholds[attack] = threshold
    scalers[attack] = scaler

    print(f"{attack} 模型训练完成，95% 阈值: {threshold:.5f}")

# ========== 5. 准备未知攻击样本（如 normal） ==========
print("\n=== 准备未知与已知攻击混合数据集 ===")
unknown_data = df[df['type'] == 'ddos'].sample(n=200, random_state=42)
X_unknown = unknown_data.select_dtypes(include=[np.number])
y_unknown_true = ['unknown'] * len(X_unknown)

# 准备已知攻击样本（每种攻击各 50 条）
known_data = pd.DataFrame()
for attack in attack_types:
    samples = df[df['type'] == attack].sample(n=200, random_state=42)
    known_data = pd.concat([known_data, samples])
X_known = known_data.select_dtypes(include=[np.number])
y_known_true = known_data['type'].tolist()

# 合并数据
X_test = pd.concat([X_unknown, X_known], ignore_index=True)
y_true = y_unknown_true + y_known_true

X_test_np = X_test.to_numpy()

# ========== 6. 批量计算每类 AE 的重构误差 ==========
print("\n=== 正在进行批量推理 ===")
mse_dict = {}
for attack in attack_types:
    scaler = scalers[attack]
    ae = autoencoders[attack]
    threshold = thresholds[attack]

    X_scaled = scaler.transform(X_test_np)
    X_recon = ae.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(X_scaled - X_recon), axis=1)
    mse_dict[attack] = mse

# ========== 7. 推理判断：是否未知攻击 ==========
mse_matrix = np.vstack([mse_dict[a] for a in attack_types]).T
results = []

for i in range(len(X_test_np)):
    mse_vector = mse_matrix[i]
    over_thresholds = [mse_vector[j] > thresholds[attack_types[j]] for j in range(len(attack_types))]
    if all(over_thresholds):
        results.append("unknown")
    else:
        min_idx = np.argmin(mse_vector)
        results.append(attack_types[min_idx])

# ========== 8. 输出预测结果与评估 ==========
from collections import Counter
print("\n预测结果分布：")
print(Counter(results))

# 分类报告
print("\n分类报告：")
print(classification_report(y_true, results))

# 混淆矩阵
print("\n混淆矩阵：")
print(confusion_matrix(y_true, results))

# ========== 10. 清理资源 ==========
from tensorflow.keras import backend as K
K.clear_session()
