from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd

# 加载已训练好的模型
autoencoder = load_model("ae_model.h5")
scaler = joblib.load("scaler.pkl")
threshold = joblib.load("threshold.pkl")


def ae_detect(df):
    # 只取数值型特征
    X = df.select_dtypes(include=[np.number])
    X_scaled = scaler.transform(X)

    # 预测重构误差
    recon = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - recon, 2), axis=1)

    # 返回检测结果（0=正常，1=异常）
    return (mse > threshold).astype(int)

df = pd.read_csv("iot_features.csv")
# 假设 df 是已经读取的新数据
results = ae_detect(df)

print("检测结果分布：")
print(pd.Series(results).value_counts())

# 0 表示正常，1 表示异常
df["AE_result"] = results
print(df.head())
