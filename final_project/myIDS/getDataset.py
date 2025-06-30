import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def get_fridge_dateset():
    normal_data_dir = "D:/毕设/TON_IoT datasets/Raw_datasets/telemtry_data/IoT_filtered_normal/"
    attack_data_dir = "D:/毕设/TON_IoT datasets/Raw_datasets/telemtry_data/IoT_filtered_normal_attack/"

    # 📥 选择要加载的文件
    normal_files = ["IoT_normal_fridge_1.csv", "IoT_normal_fridge_2.csv", "IoT_normal_fridge_3.csv"]
    attack_files = ["IoT_backdoor_normal/IoT_backdoor_normal_fridge.csv",
                    "IoT_DDoS_normal/IoT_DDoS_normal_fridge.csv",
                    "IoT_DoS_normal/IoT_DoS_normal_fridge.csv",
                    "IoT_Injection_normal/IoT_Injection_normal_fridge.csv",
                    "IoT_MITM_normal/IoT_MITM_normal_fridge.csv",
                    "IoT_password_normal/IoT_password_normal_fridge.csv",
                    "IoT_runsomware_normal/IoT_runsomware_normal_fridge.csv",
                    "IoT_scanning_normal/IoT_scanning_normal_fridge1.csv",
                    "IoT_XSS_normal/IoT_XSS_normal_fridge.csv"]

    # read normal data
    df_normal = pd.concat(
        [pd.read_csv(os.path.join(normal_data_dir, f), delimiter=",", encoding="utf-8") for f in normal_files],
        ignore_index=True)
    df_normal["label"] = 0

    # read attack data
    df_list = []
    for f in attack_files:
        file_path = os.path.join(attack_data_dir, f)
        df_temp = pd.read_csv(file_path, delimiter=",", encoding="utf-8")
        # add different label by different type of attack
        if "_backdoor_" in f.lower():
            df_temp["label"] = 1
        elif "_ddos_" in f.lower():
            df_temp["label"] = 2
        elif "_dos_" in f.lower():
            df_temp["label"] = 3
        elif "_injection_" in f.lower():
            df_temp["label"] = 4
        elif "_mitm_" in f.lower():
            df_temp["label"] = 5
        elif "_password_" in f.lower():
            df_temp["label"] = 6
        elif "_runsomware_" in f.lower():
            df_temp["label"] = 7
        elif "_scanning_" in f.lower():
            df_temp["label"] = 8
        elif "_xss_" in f.lower():
            df_temp["label"] = 9
        else:
            df_temp["label"] = 0
        df_list.append(df_temp)
        print(f"{df_temp.columns},{df_temp.shape}")
        df_temp.head(5)

    df_attack = pd.concat(df_list, ignore_index=True)
    df_attack.rename(columns={"date": "Date"}, inplace=True)
    df_attack.rename(columns={"time": "Time"}, inplace=True)

    data = pd.concat([df_normal, df_attack], ignore_index=True)
    print(f"Size of the merged data：{data.shape}")
    label_counts = data["label"].value_counts()
    print(label_counts)
    count_label_0 = (data["label"] == 0).sum()
    count_label_other = (data["label"] != 0).sum()
    print(f"normal data: {count_label_0}")
    print(f"attack data: {count_label_other}")

    # Processing time characteristic
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d-%b-%y %H:%M:%S', errors='coerce')
    data['Hour'] = data['Datetime'].dt.hour
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek

    data.drop(columns=['Date', 'Time', 'Datetime'], inplace=True)

    # Processing label feature
    label_encoder = LabelEncoder()
    data['Temp_Condition'] = label_encoder.fit_transform(data['Temp_Condition'])

    # choose useful feature
    features = ['Fridge_Temperature', 'Temp_Condition', 'Hour', 'DayOfWeek']
    X = data[features]
    y = data['label']

    # Divide the training set and test set
    # X = df.drop(columns=["label","Unnamed: 4","Unnamed: 5","Unnamed: 6","Unnamed: 7","Unnamed: 8","Unnamed: 9","Unnamed: 10"])  # 特征
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X, y
