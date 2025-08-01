import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime


def get_fridge_dateset():
    data_dir = "D:/毕设/TON_IoT datasets/Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Fridge.csv"

    # read normal data
    data = pd.read_csv(os.path.join(data_dir), delimiter=",", encoding="utf-8")
    # data = data.iloc[:10000]
    data['date'] = data['date'].str.strip()
    data['time'] = data['time'].str.strip()
    data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'], format="%d-%b-%y %H:%M:%S",
                                            errors='coerce')

    print(f"attack data shape:{data.columns},{data.shape}")
    print(data.head(5))
    count_label_0 = (data["label"] == 0).sum()
    count_label_other = (data["label"] != 0).sum()
    print(f"normal data: {count_label_0}")
    print(f"attack data: {count_label_other}")

    # Processing time characteristic
    data = data.dropna(subset=['timestamp'])
    # data = data.sort_values(by='timestamp')
    print(len(data))
    # Processing label feature
    label_encoder = LabelEncoder()
    data['temp_condition'] = label_encoder.fit_transform(data['temp_condition'])

    label_counts = data["label"].value_counts()
    print(label_counts)
    # data = data.sort_values('timestamp').reset_index(drop=True)

    # choose useful feature
    feature_cols = ['fridge_temperature', 'temp_condition']
    label_col = 'label'

    scaler = MinMaxScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])

    # 窗口长度（秒）
    window_size = 60
    # 生成样本：每个窗口内的数据作为一个样本，标签取窗口内最后一个时刻
    sequences = []
    labels = []

    start_time = data['timestamp'].min()
    end_time = data['timestamp'].max()

    window_delta = pd.Timedelta(seconds=window_size)
    t = start_time

    while t + window_delta <= end_time:
        window_df = data[(data['timestamp'] >= t) & (data['timestamp'] < t + window_delta)]
        if len(window_df) == 0:
            t += pd.Timedelta(seconds=1)
            continue
        if len(window_df) < 5:  # 你可以设定最小长度门槛
            t += pd.Timedelta(seconds=1)
            continue

        seq = window_df[feature_cols].values
        counts = np.bincount(window_df[label_col])
        label = np.argmax(counts)
        # label = window_df[label_col].values[-1]  # 取最后一个标签或多数投票
        sequences.append(seq)
        labels.append(label)
        t += pd.Timedelta(seconds=1)

    max_len = max([len(s) for s in sequences])
    X = pad_sequences(sequences, maxlen=max_len, dtype='float32', padding='post')
    y = np.array(labels)
    print("标签分布:", np.bincount(y))
    print("数据总行数:", len(data))
    print("开始时间:", start_time)
    print("结束时间:", end_time)
    print("所有 label 分布:", data['label'].value_counts())
    print("label=0 的时间范围:", data[data['label'] == 0]['timestamp'].min(), " ~ ",
          data[data['label'] == 0]['timestamp'].max())
    print("label=1 的时间范围:", data[data['label'] == 1]['timestamp'].min(), " ~ ",
          data[data['label'] == 1]['timestamp'].max())

    return X, y
