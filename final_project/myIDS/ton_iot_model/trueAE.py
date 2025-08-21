import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score
import joblib

# Data preprocessing
df = pd.read_csv("csvdataset/train1.csv")
y = df['label'].values

drop_cols = ['src_ip', 'dst_ip', 'http_uri', 'http_user_agent', 'type']
X = df.drop(columns=drop_cols + ['label'])

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

X[cat_cols] = X[cat_cols].fillna('NaN')
X[num_cols] = X[num_cols].fillna(0)

preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
X_processed = preprocessor.fit_transform(X)
X_processed = X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed
print("Feature-dimension after processing:", X_processed.shape)

# Autoencoder model definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=16, dropout_num=0.05):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Dropout(dropout_num),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(dropout_num),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout_num),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout_num),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(dropout_num),
            nn.Linear(32, bottleneck_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32), nn.ReLU(),
            nn.Dropout(dropout_num),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Dropout(dropout_num),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Dropout(dropout_num),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Dropout(dropout_num),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Dropout(dropout_num),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Training preparation
X_train = X_processed[y == 0]
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train_tensor, X_train_tensor), batch_size=128, shuffle=True)

input_dim = X_processed.shape[1]
model = Autoencoder(input_dim, bottleneck_dim=8, dropout_num=0.05)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Model training
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, _ in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_X)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}")

# Inference & Threshold Search
X_all_tensor = torch.tensor(X_processed, dtype=torch.float32)
model.eval()
with torch.no_grad():
    reconstructed = model(X_all_tensor)
    errors = torch.mean((X_all_tensor - reconstructed) ** 2, dim=1).numpy()

# Preliminary search for thresholds
best_f1, best_th = 0, None
th_candidates = np.linspace(errors.min(), errors.max(), 2000)
for th in th_candidates:
    y_pred = (errors > th).astype(int)
    f1 = f1_score(y, y_pred, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

# Find thresholds for more detail
fine_range = np.linspace(best_th - 1e-6, best_th + 1e-6, 10000)
for th in fine_range:
    y_pred = (errors > th).astype(int)
    f1 = f1_score(y, y_pred, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"best threshold: {best_th:.8f}, corresponding Macro-F1: {best_f1:.6f}")

# Save the model and thresholds
torch.save(model.state_dict(), "deploy_ae/best_autoencoder_model.pth")
with open("deploy_ae/best_model_info.json", "w") as f:
    json.dump({"best_threshold": float(best_th), "macro_f1": float(best_f1)}, f)
print("model and threshold has been save")

joblib.dump(preprocessor, "deploy_ae/ae_preprocessor.pkl")
joblib.dump(X.columns.tolist(), "deploy_ae/ae_feature_names.pkl")

y_pred = (errors > best_th).astype(int)
print(classification_report(y, y_pred))