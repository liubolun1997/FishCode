import pandas as pd
from sklearn.model_selection import train_test_split

df_1 = pd.read_csv("./csvdataset/UNSW_2018_IoT_Botnet_Full5pc_dos&ddos.csv")
df_2 = pd.read_csv("./csvdataset/UNSW_2018_IoT_Botnet_Full5pc_dos.csv")
df_3 = pd.read_csv("./csvdataset/UNSW_2018_IoT_Botnet_Full5pc_ddos.csv")
df_4 = pd.read_csv("./csvdataset/UNSW_2018_IoT_Botnet_Full5pc_4.csv")

all_df = pd.concat([df_1, df_2, df_3, df_4],ignore_index=True)
print(all_df['category'].value_counts())

# Drop Normal and Theft
df_attack = all_df[~all_df['category'].isin(['Normal','Theft'])]

# Downsampling
ddos_sample = df_attack[df_attack['category']=='DDoS'].sample(n=100000, random_state=42)
dos_sample = df_attack[df_attack['category']=='DoS'].sample(n=100000, random_state=42)
recon_sample = df_attack[df_attack['category']=='Reconnaissance']
# merge
balanced_df = pd.concat([ddos_sample, dos_sample, recon_sample])
# save
balanced_df.to_csv("./csvdataset/Bot-IoT_balanced.csv", index=False)
print(balanced_df['category'].value_counts())

# Split the 15% validation set by label tier
validation_fraction = 0.15
train_df, val_df = train_test_split(
    balanced_df,
    test_size=validation_fraction,
    stratify=balanced_df['category'],
    random_state=42
)

# Sort by type
train_df = train_df.sort_values(by='category').reset_index(drop=True)
val_df = val_df.sort_values(by='category').reset_index(drop=True)

# save .csv
train_df.to_csv("./csvdataset/train1.csv", index=False)
val_df.to_csv("./csvdataset/validation1.csv", index=False)

print(f"Number of training set samples: {len(train_df)}")
print(f"Validation set sample size: {len(val_df)}")

print("\nTraining set type distribution:")
print(train_df['category'].value_counts())
print("\nValidate the set type distribution:")
print(val_df['category'].value_counts())