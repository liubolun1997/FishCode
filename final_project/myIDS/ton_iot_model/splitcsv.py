import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("D:/毕设/TON_IoT datasets/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv")
# df = pd.read_csv("./iot_features_mapped.csv")
# df = pd.concat([df_2,df_1],ignore_index=True)

# Split the 15% validation set by label tier

validation_fraction = 0.15
train_df, val_df = train_test_split(
    df,
    test_size=validation_fraction,
    stratify=df['label'],
    random_state=42
)

# Sort by type
train_df = train_df.sort_values(by='type').reset_index(drop=True)
val_df = val_df.sort_values(by='type').reset_index(drop=True)

# Output CSV file
train_df.to_csv("./csvdataset/train1.csv", index=False)
val_df.to_csv("./csvdataset/validation1.csv", index=False)

print(f"Number of training set samples: {len(train_df)}")
print(f"Validation set sample size: {len(val_df)}")

print("\nTraining set type distribution:")
print(train_df['type'].value_counts())
print("\nValidate the set type distribution:")
print(val_df['type'].value_counts())
