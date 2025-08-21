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

# loading data
df = pd.read_csv('./csvdataset/train1.csv').dropna()
# Select the known vs. unknown attack category
# Known attack category('DoS','DDoS')
known_categories = ['DoS','DDoS']
# Unknown attack category
unknown_categories = ['Reconnaissance']

# Training set: Known attacks
train_df_all, test_df_all = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df['category']
)
train_df = train_df_all[(train_df_all['category'].isin(known_categories))]

# Features and tags
drop_cols = [
    'pkSeqID','stime','ltime','saddr','sport','daddr','dport',
    'category','subcategory','attack'
]
X = train_df.drop(columns=drop_cols)
y = train_df['category']

# Handle taxonomic features
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

X[cat_cols] = X[cat_cols].fillna('NaN')
X[num_cols] = X[num_cols].fillna(0)

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
joblib.dump(encoders, "./deploy_xgb/encoders_xgb.pkl")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "./deploy_xgb/label_encoder_xgb.pkl")

# standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "./deploy_xgb/scaler_xgb.pkl")

# Train XGBoost
xgb = XGBClassifier(eval_metric='mlogloss')
xgb.fit(X_train_scaled, y_encoded)
xgb.save_model("./deploy_xgb/xgb_model.json")

# Confidence threshold
train_probs = xgb.predict_proba(X_train_scaled)
train_max_proba = np.max(train_probs, axis=1)
confidence_threshold = np.percentile(train_max_proba, 5)  # 5% quantile
np.save("./deploy_xgb/threshold_xgb.npy", confidence_threshold)
print(f"Dynamic threshold = {confidence_threshold:.4f}")

# Construct Test Set (Known + Unknown)
X_test = test_df_all.drop(columns=drop_cols)

# Fill in the missing values
X_test[cat_cols] = X_test[cat_cols].fillna('NaN')
X_test[num_cols] = X_test[num_cols].fillna(0)

# Convert Category Encoding (Unknown Category → -1)
for col in cat_cols:
    le = encoders[col]
    X_test[col] = X_test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

X_test_scaled = scaler.transform(X_test)

# True labels: Known=0, Unknown=1
true_labels = test_df_all['category'].apply(lambda t: 1 if t in unknown_categories else 0).values

# forecast
probs = xgb.predict_proba(X_test_scaled)
max_probs = np.max(probs, axis=1)
pred_labels = (max_probs < confidence_threshold).astype(int)

# assess
print("\n=== Confusion Matrix (Known=0, Unknown=1) ===")
print(confusion_matrix(true_labels, pred_labels))
print("\n=== Performance ===")
print(f"Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")
print(classification_report(true_labels, pred_labels, target_names=['Known','Unknown']))
