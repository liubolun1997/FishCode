import pandas as pd
import numpy as np
import json
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import time
import psutil, os

# TensorFlow CPU-only setting
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Keep the original AE model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=8, dropout_num=0.05):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout_num),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_num),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_num),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout_num),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout_num),
            nn.Linear(32, bottleneck_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32), nn.ReLU(), nn.Dropout(dropout_num),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(dropout_num),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(dropout_num),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout_num),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(dropout_num),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load data
df = pd.read_csv("csvdataset/val_stream2.csv")
df = df[~df['type'].isin(['mitm'])]
drop_cols = ['src_ip', 'dst_ip', 'http_uri', 'http_user_agent', 'type']

cat_cols = df.drop(columns=drop_cols + ['label']).select_dtypes(include=['object']).columns.tolist()
num_cols = df.drop(columns=drop_cols + ['label']).select_dtypes(exclude=['object']).columns.tolist()

df[cat_cols] = df[cat_cols].fillna('NaN')
df[num_cols] = df[num_cols].fillna(0)

# Performance measurement begins
begin_time = time.time()

# Lazy loading AEs
print("Loading Autoencoder...")
with open("deploy_ae/best_model_info.json", "r") as f:
    ae_info = json.load(f)
ae_threshold = ae_info["best_threshold"]
preprocessor = joblib.load("deploy_ae/ae_preprocessor.pkl")
ae_feature_names = joblib.load("deploy_ae/ae_feature_names.pkl")

X_processed = preprocessor.transform(df).toarray()
input_dim = X_processed.shape[1]

ae_model = Autoencoder(input_dim, bottleneck_dim=8)
ae_model.load_state_dict(torch.load("deploy_ae/best_autoencoder_model.pth", map_location='cpu'))
ae_model.eval()

# Batch AE
batch_size = 64
X_tensor = torch.tensor(X_processed, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=batch_size)

errors_list = []
with torch.no_grad():
    for batch in loader:
        batch_X = batch[0]
        reconstructed = ae_model(batch_X)
        batch_errors = torch.mean((batch_X - reconstructed)**2, dim=1)
        errors_list.append(batch_errors)

errors = torch.cat(errors_list).numpy()
df["ae_pred"] = (errors > ae_threshold).astype(int)

# Lazy loading XGBoost
attack_df = df[df["ae_pred"] == 1].copy()
if not attack_df.empty:
    print("Loading XGBoost...")
    xgb_model = XGBClassifier()
    xgb_model.load_model("deploy_xgb/xgb_model.json")
    xgb_threshold = np.load("deploy_xgb/threshold_xgb.npy")
    xgb_encoders = joblib.load("deploy_xgb/encoders_xgb.pkl")
    xgb_scaler = joblib.load("deploy_xgb/scaler_xgb.pkl")

    X_attack = attack_df.drop(columns=drop_cols + ['label', 'ae_pred'])
    cat_cols = X_attack.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_attack.select_dtypes(exclude=['object']).columns.tolist()

    X_attack[cat_cols] = X_attack[cat_cols].fillna('NaN')
    X_attack[num_cols] = X_attack[num_cols].fillna(0)

    for col in cat_cols:
        le_col = xgb_encoders[col]
        X_attack[col] = X_attack[col].map(
            lambda s: le_col.transform([s])[0] if s in le_col.classes_ else -1
        )

    # Batch XGBoost
    X_scaled = xgb_scaler.transform(X_attack)
    probs_list = []
    for i in range(0, len(X_scaled), batch_size):
        probs_list.append(xgb_model.predict_proba(X_scaled[i:i+batch_size]))
    probs = np.vstack(probs_list)
    max_probs = np.max(probs, axis=1)
    attack_df["xgb_pred_model"] = (max_probs < xgb_threshold).astype(int)
    df["xgb_pred"] = None
    df.loc[attack_df.index, "xgb_pred"] = attack_df["xgb_pred_model"]

# Lazy loading RNNs
known_attack_df = attack_df[attack_df["xgb_pred_model"] == 0] if not attack_df.empty else pd.DataFrame()
if not known_attack_df.empty:
    print("Loading RNN...")
    rnn_model = load_model("deploy_rnn/rnn_bilstm_model.h5")
    label_encoder_rnn = joblib.load("deploy_rnn/label_encoder_rnn.pkl")
    rnn_scaler = joblib.load("deploy_rnn/scaler_rnn.pkl")
    encoders_rnn = joblib.load("deploy_rnn/encoders_rnn.pkl")

    X_known = known_attack_df.drop(columns=drop_cols + ['label', 'ae_pred', 'xgb_pred_model'])
    cat_cols_known = X_known.select_dtypes(include=['object']).columns.tolist()
    num_cols_known = X_known.select_dtypes(exclude=['object']).columns.tolist()

    X_known[cat_cols_known] = X_known[cat_cols_known].fillna('NaN')
    X_known[num_cols_known] = X_known[num_cols_known].fillna(0)

    for col in cat_cols_known:
        le_col = encoders_rnn[col]
        X_known[col] = X_known[col].map(lambda s: le_col.transform([s])[0] if s in le_col.classes_ else -1)

    X_known_scaled = rnn_scaler.transform(X_known)
    time_steps = 5
    X_seq_list = [X_known_scaled[i:i+time_steps] for i in range(len(X_known_scaled)-time_steps)]
    rnn_preds_list = []
    for i in range(0, len(X_seq_list), batch_size):
        batch_seq = np.array(X_seq_list[i:i+batch_size])
        rnn_preds_list.append(rnn_model.predict(batch_seq, verbose=0))
    rnn_preds = np.vstack(rnn_preds_list)
    rnn_labels = label_encoder_rnn.inverse_transform(np.argmax(rnn_preds, axis=1))

    known_attack_df = known_attack_df.iloc[time_steps:].copy()
    known_attack_df["rnn_pred"] = rnn_labels
    df["rnn_pred"] = pd.Series(dtype="object")
    df.loc[known_attack_df.index, "rnn_pred"] = known_attack_df["rnn_pred"]

# Construct the final prediction result column ids_pred
df["ids_pred"] = 'NaN'
df.loc[df["ae_pred"] == 0, "ids_pred"] = "normal"
if "xgb_pred" in df.columns:
    df.loc[(df["ae_pred"] == 1) & (df["xgb_pred"] == 1), "ids_pred"] = "unknown"
if "rnn_pred" in df.columns:
    df.loc[df["rnn_pred"].notnull(), "ids_pred"] = df["rnn_pred"]
    df.loc[df["rnn_pred"] == "normal", "ids_pred"] = "normal"

df["real_type"] = df["type"].copy()
df.loc[df["type"] == "password", "real_type"] = "unknown"

# The model reasoning ends
end_time = time.time()
elapsed_time = end_time - begin_time

print("\n[Autoencoder model] binary classification accuracy:")
print(classification_report(df["real_type"].apply(lambda x: "Attack" if x != "normal" else "Normal"),
                            df["ae_pred"].apply(lambda x: "Attack" if x == 1 else "Normal")))

if "xgb_pred" in df.columns:
    xgb_eval_df = df[(df["ae_pred"] == 1) & (df["real_type"] != "normal")]
    print("\n[XGBoost model] known vs. unknown attack accuracy:")
    print(classification_report(
        xgb_eval_df["real_type"].apply(lambda x: "unknown" if x == "unknown" else "known"),
        xgb_eval_df["xgb_pred"].apply(lambda x: "unknown" if x == 1 else "known")
    ))

eval_df = df.dropna(subset=["real_type", "ids_pred"]).copy()
labels = sorted([lbl for lbl in eval_df["real_type"].unique() if lbl not in ["password",'NaN']])
print("\n[Overall IDS Framework] Final prediction accuracy after fusing three layers:")
print(classification_report(eval_df["real_type"], eval_df["ids_pred"], labels=labels))
print("Confusion matrix:")
print(confusion_matrix(eval_df["real_type"], eval_df["ids_pred"],labels=labels))
# End of performance measurement
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
