import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

# 确保保存目录存在
os.makedirs("deploy_xgb", exist_ok=True)

# 1. 加载数据并筛选
df = pd.read_csv('D:/毕设/TON_IoT datasets/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv')
df = df.dropna()

train_df = df[~df['type'].isin(['xss','injection','password'])]

drop_cols = ['src_ip', 'dst_ip', 'http_uri', 'http_user_agent', 'type']

X = train_df.drop(columns=drop_cols + ['label','type'])
y = train_df['type']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

X[cat_cols] = X[cat_cols].fillna('NaN')
X[num_cols] = X[num_cols].fillna(0)

# Encoders that hold the characteristics of each category
encoders = {}
for col in cat_cols:
    le_col = LabelEncoder()
    X[col] = le_col.fit_transform(X[col])
    encoders[col] = le_col

joblib.dump(encoders, "deploy_xgb/encoders_xgb.pkl")

# label encoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "deploy_xgb/label_encoder_xgb.pkl")

# split trainingset and testset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# standardscaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "deploy_xgb/scaler_xgb.pkl")

# training XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train_scaled, y_train)

# seve model
xgb.save_model("deploy_xgb/xgb_model.json")

# ============ get threshold ============
train_probs = xgb.predict_proba(X_train_scaled)
train_max_proba = np.max(train_probs, axis=1)
confidence_threshold = np.percentile(train_max_proba, 5)
np.save("deploy_xgb/threshold_xgb.npy", confidence_threshold)
print(f"The threshold set dynamically is: {confidence_threshold:.4f}")

# ============ 测试阶段（已知 vs 未知攻击） ============
known_attack_df = df[df['type'].isin(['dos', 'backdoor', 'ddos', 'mitm','scanning','ransomware'])].sample(n=100000, random_state=42)
unknown_attack_df = df[df['type'] == 'password'].sample(n=10000, random_state=42)

test_df = pd.concat([known_attack_df, unknown_attack_df])
X_test_all = test_df.drop(columns=drop_cols + ['label','type'])

X_test_all[cat_cols] = X_test_all[cat_cols].fillna('NaN')
X_test_all[num_cols] = X_test_all[num_cols].fillna(0)

# Convert with a saved encoder
for col in cat_cols:
    le_col = encoders[col]
    X_test_all[col] = X_test_all[col].map(lambda s: le_col.transform([s])[0] if s in le_col.classes_ else -1)

X_test_scaled = scaler.transform(X_test_all)

true_labels = np.array([0]*len(known_attack_df) + [1]*len(unknown_attack_df))
probs = xgb.predict_proba(X_test_scaled)
max_probs = np.max(probs, axis=1)
predicted_labels = (max_probs < confidence_threshold).astype(int)

# Evaluate the results
print("\n=== Confusion matrix ===")
cm = confusion_matrix(true_labels, predicted_labels)
print(cm)
print("\n=== Performance metrics ===")
print(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
print(classification_report(true_labels, predicted_labels,
                           target_names=['known', 'unknown']))
