import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import time
import psutil, os
from sklearn.metrics import accuracy_score

# Load the validation set
df = pd.read_csv("csvdataset/val_stream1.csv")

# List irrelevant features
drop_cols = [
    'pkSeqID','stime','ltime','saddr','sport','daddr','dport',
    'category','subcategory','attack'
]

cat_cols = df.drop(columns=drop_cols).select_dtypes(include=['object']).columns.tolist()
num_cols = df.drop(columns=drop_cols).select_dtypes(exclude=['object']).columns.tolist()

# Fill in the missing values
df[cat_cols] = df[cat_cols].fillna('NaN')
df[num_cols] = df[num_cols].fillna(0)

# Performance measurement begin
begin_time = time.time()

# XGBoost module inference
print("Loading XGBoost...")
xgb_model = XGBClassifier()
xgb_model.load_model("deploy_xgb/xgb_model.json")
xgb_threshold = np.load("deploy_xgb/threshold_xgb.npy")
xgb_encoders = joblib.load("deploy_xgb/encoders_xgb.pkl")
xgb_scaler = joblib.load("deploy_xgb/scaler_xgb.pkl")

X_xgb = df.drop(columns=drop_cols)
for col in cat_cols:
    le_col = xgb_encoders[col]
    X_xgb[col] = X_xgb[col].map(lambda s: le_col.transform([s])[0] if s in le_col.classes_ else -1)

# transform data
X_xgb_scaled = xgb_scaler.transform(X_xgb)
probs = xgb_model.predict_proba(X_xgb_scaled)
max_probs = np.max(probs, axis=1)
df["xgb_pred"] = (max_probs < xgb_threshold).astype(int)  # 1 = unknown, 0 = known

# RNN-BiLSTM module inference
# Only RNN classification is performed for attacks that XGBoost determines to be known
known_attack_df = df[df["xgb_pred"] == 0].copy()
if not known_attack_df.empty:
    print("Loading RNN...")
    rnn_model = load_model("deploy_rnn/rnn_bilstm_model.h5")
    label_encoder_rnn = joblib.load("deploy_rnn/label_encoder_rnn.pkl")
    rnn_scaler = joblib.load("deploy_rnn/scaler_rnn.pkl")
    encoders_rnn = joblib.load("deploy_rnn/encoders_rnn.pkl")

    X_rnn = known_attack_df.drop(columns=drop_cols + ['xgb_pred'])
    cat_cols_rnn = X_rnn.select_dtypes(include=['object']).columns.tolist()
    num_cols_rnn = X_rnn.select_dtypes(exclude=['object']).columns.tolist()

    X_rnn[cat_cols_rnn] = X_rnn[cat_cols_rnn].fillna('NaN')
    X_rnn[num_cols_rnn] = X_rnn[num_cols_rnn].fillna(0)

    for col in cat_cols_rnn:
        le_col = encoders_rnn[col]
        X_rnn[col] = X_rnn[col].map(lambda s: le_col.transform([s])[0] if s in le_col.classes_ else -1)

    X_rnn_scaled = rnn_scaler.transform(X_rnn)
    time_steps = 5
    X_seq_list = [X_rnn_scaled[i:i+time_steps] for i in range(len(X_rnn_scaled)-time_steps)]
    rnn_preds_list = []
    # Set up batch processing
    batch_size = 64
    for i in range(0, len(X_seq_list), batch_size):
        batch_seq = np.array(X_seq_list[i:i+batch_size])
        rnn_preds_list.append(rnn_model.predict(batch_seq, verbose=0))
    rnn_preds = np.vstack(rnn_preds_list)
    rnn_labels = label_encoder_rnn.inverse_transform(np.argmax(rnn_preds, axis=1))

    known_attack_df = known_attack_df.iloc[time_steps:].copy()
    known_attack_df["rnn_pred"] = rnn_labels
    df["rnn_pred"] = pd.Series(dtype="object")
    df.loc[known_attack_df.index, "rnn_pred"] = known_attack_df["rnn_pred"]

# Construct the final prediction result ids_pred
df["ids_pred"] = 'unknown'
if "rnn_pred" in df.columns:
    df.loc[df["rnn_pred"].notnull(), "ids_pred"] = df["rnn_pred"]

# Handle unknown label mapping
df["real_type"] = df["category"].copy()
df.loc[df["category"] == "Reconnaissance", "real_type"] = "unknown"

# The model reasoning ends
end_time = time.time()
elapsed_time = end_time - begin_time

# assess
# XGBoost accuracy
df["xgb_true"] = df["category"].apply(lambda x: "unknown" if x=="Reconnaissance" else "known")
xgb_true_labels = df["xgb_true"]
xgb_pred_labels = df["xgb_pred"].map({0: "known", 1: "unknown"})
print("\n[XGBoost model] known vs unknown attack accuracy:")
print(f"Accuracy: {accuracy_score(xgb_true_labels, xgb_pred_labels):.4f}")
print(classification_report(xgb_true_labels, xgb_pred_labels, target_names=["known","unknown"]))
print("Confusion matrix:")
print(confusion_matrix(xgb_true_labels, xgb_pred_labels, labels=["known","unknown"]))

# RNN-BiLSTM accuracy
if not known_attack_df.empty:
    rnn_true_labels = known_attack_df["category"]
    rnn_pred_labels = known_attack_df["rnn_pred"]
    print("\n[RNN-BiLSTM model] multi-class attack classification accuracy:")
    print(f"Accuracy: {accuracy_score(rnn_true_labels, rnn_pred_labels):.4f}")
    print(classification_report(rnn_true_labels, rnn_pred_labels))
    print("Confusion matrix:")
    print(confusion_matrix(rnn_true_labels, rnn_pred_labels, labels=rnn_true_labels.unique()))


eval_df = df.dropna(subset=["real_type", "ids_pred"]).copy()
labels = sorted([lbl for lbl in eval_df["real_type"].unique() if lbl not in ["NaN"]])
print("\n[Overall IDS Framework] Final prediction accuracy:")
print(classification_report(eval_df["real_type"], eval_df["ids_pred"], labels=labels))
print("Confusion matrix:")
print(confusion_matrix(eval_df["real_type"], eval_df["ids_pred"], labels=labels))

# Performance metrics
n_samples = len(eval_df)
latency = (elapsed_time / n_samples) * 1000 if n_samples > 0 else 0.0
throughput = n_samples / elapsed_time if elapsed_time > 0 else 0.0
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / 1024**2
cpu_usage = psutil.cpu_percent(interval=1)

print("\n[Performance metric measurement]")
print(f"Total reasoning time: {elapsed_time:.4f} second")
print(f"Average inference delay: {latency:.4f} ms/sample")
print(f"throughput: {throughput:.2f} samples/sec")
print(f"Memory Usage: {memory_usage:.2f} MB")
print(f"CPU utilization: {cpu_usage:.2f} %")
