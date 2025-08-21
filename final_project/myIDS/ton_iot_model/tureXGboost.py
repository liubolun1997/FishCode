import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

# Make sure the save directory exists
os.makedirs("deploy_xgb", exist_ok=True)

# Load data and filter
df_all = pd.read_csv('csvdataset/train1.csv')
df_all = df_all.dropna()
df_all = df_all[~df_all['type'].isin(['mitm'])]
df, df_test= train_test_split(df_all, test_size=0.2, random_state=42)
unknown_attack = ['password']

train_df = df[~df['type'].isin(unknown_attack)]
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

# standardscaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "deploy_xgb/scaler_xgb.pkl")

# training XGBoost model
xgb = XGBClassifier(eval_metric='mlogloss')
xgb.fit(X_train_scaled, y_encoded)

# seve model
xgb.save_model("deploy_xgb/xgb_model.json")

# get threshold
train_probs = xgb.predict_proba(X_train_scaled)
train_max_proba = np.max(train_probs, axis=1)
confidence_threshold = np.percentile(train_max_proba, 5)
np.save("deploy_xgb/threshold_xgb.npy", confidence_threshold)
print(f"The threshold set dynamically is: {confidence_threshold:.4f}")

# Testing phase (known vs. unknown attacks)
known_attack_df = df_test[~df_test['type'].isin(unknown_attack)]
unknown_attack_df = df_test[df_test['type'].isin(unknown_attack)]

X_test_all = df_test.drop(columns=drop_cols + ['label','type'])

X_test_all[cat_cols] = X_test_all[cat_cols].fillna('NaN')
X_test_all[num_cols] = X_test_all[num_cols].fillna(0)

# Convert with a saved encoder
for col in cat_cols:
    le_col = encoders[col]
    X_test_all[col] = X_test_all[col].map(lambda s: le_col.transform([s])[0] if s in le_col.classes_ else -1)

X_test_scaled = scaler.transform(X_test_all)

true_labels = df_test['type'].apply(lambda t: 1 if t in unknown_attack else 0).values
probs = xgb.predict_proba(X_test_scaled)
max_probs = np.max(probs, axis=1)
predicted_labels = (max_probs < confidence_threshold).astype(int)

# ====== Evaluate the results ========
print("\n=== Confusion matrix ===")
cm = confusion_matrix(true_labels, predicted_labels)
print(cm)
print("\n=== Performance metrics ===")
print(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
print(classification_report(true_labels, predicted_labels,
                           target_names=['known', 'unknown']))
