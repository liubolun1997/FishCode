import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

# ========== 1. 加载和预处理数据 ==========
# 替换为你的文件路径
df = pd.read_csv('D:/毕设/TON_IoT datasets/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv')
df = df.dropna()
df = df[df['type'].isin(['backdoor','ddos'])]
# 识别标签列（你可能需要根据实际情况调整）
label_col = 'label' if 'label' in df.columns else 'Attack_type'

# 查看类别有哪些
print("标签分布：")
print(df[label_col].value_counts())

# 仅保留数值特征
X = df.select_dtypes(include=[np.number])
y = df['type']

# 根据实际正常标签名称调整（TON_IoT通常为 "Normal" 或 "BENIGN"）
# normal_mask = y.str.lower() == "normal"
normal_mask = y == 'backdoor'

X_normal = X[normal_mask]
X_anomalous = X[~normal_mask]

# ========== 2. 划分训练集（只使用正常数据） ==========
X_train, X_val = train_test_split(X_normal, test_size=0.25, random_state=42)

# 标准化（仅用正常样本 fit）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_all_scaled = scaler.transform(X)

# ========== 3. 构建 Autoencoder 模型 ==========
input_dim = X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,))

encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu',
                activity_regularizer=regularizers.l1(1e-5))(encoded)
decoded = Dense(64, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# ========== 4. 模型训练 ==========
history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(X_val_scaled, X_val_scaled),
    verbose=1
)

# ========== 5. 重构误差计算 ==========
reconstructions = autoencoder.predict(X_all_scaled)
mse = np.mean(np.power(X_all_scaled - reconstructions, 2), axis=1)

# 阈值选择（可使用验证集调优，默认 0.1）
threshold = 0.01
print(f"使用阈值：{threshold:.3f}")

# ========== 6. 评估 ==========
y_true = np.where(normal_mask, 0, 1)  # 0: 正常，1: 攻击
y_pred = (mse > threshold).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
