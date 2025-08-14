import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 加载数据 =====================
df = pd.read_csv('D:/毕设/TON_IoT datasets/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv')
df = df.dropna()

# 仅保留已知攻击类型
attack_types = ['injection', 'xss']
df = df[~df['type'].isin(attack_types)]

# ===================== 2. 特征与标签 =====================
drop_cols = ['src_ip', 'dst_ip', 'http_uri', 'http_user_agent', 'type']

X = df.drop(columns=drop_cols + ['label','type'])
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

X[cat_cols] = X[cat_cols].fillna('NaN')
X[num_cols] = X[num_cols].fillna(0)

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# X = df.select_dtypes(include=[np.number])  # 仅使用数值型字段
y = df['type']

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# 保存标签编码器
joblib.dump(label_encoder, "deploy_rnn/label_encoder_rnn.pkl")

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "deploy_rnn/scaler_rnn.pkl")  # 保存标准化器

# ===================== 3. 构造序列数据 =====================
time_steps = 5
n_samples = len(X_scaled) - time_steps
X_seq = np.array([X_scaled[i:i+time_steps] for i in range(n_samples)])
y_seq = y_onehot[time_steps:]

print(f"输入维度: {X_seq.shape}, 标签维度: {y_seq.shape}")

# ===================== 4. 划分训练/测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# ===================== 5. 构建 RNN + BiLSTM 模型 =====================
model = Sequential()
model.add(SimpleRNN(64, return_sequences=True, input_shape=(time_steps, X_seq.shape[2])))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_seq.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===================== 6. 训练模型 =====================
history = model.fit(X_train, y_train, epochs=20, batch_size=64,
                    validation_data=(X_test, y_test), verbose=1)

# 保存模型（HDF5 格式）
model.save("deploy_rnn/rnn_bilstm_model.h5")

# ===================== 7. 模型评估 =====================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n 测试准确率: {accuracy * 100:.2f}%")

# 预测结果
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# ===================== 8. 输出报告 =====================
print("\n 分类报告:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

print(" 混淆矩阵:")
print(confusion_matrix(y_true, y_pred))
