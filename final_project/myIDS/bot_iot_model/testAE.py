import joblib
import numpy as np
import pandas as pd


# df_1 = pd.read_csv("./csvdataset/UNSW_2018_IoT_Botnet_Full5pc_dos&ddos.csv")
# df_2 = pd.read_csv("./csvdataset/UNSW_2018_IoT_Botnet_Full5pc_dos.csv")
# df_3 = pd.read_csv("./csvdataset/UNSW_2018_IoT_Botnet_Full5pc_ddos.csv")
# df_4 = pd.read_csv("./csvdataset/UNSW_2018_IoT_Botnet_Full5pc_4.csv")
#
# all_df = pd.concat([df_1, df_2, df_3, df_4],ignore_index=True)
# print(all_df['category'].value_counts())

df_2 = pd.read_csv("./csvdataset/val_stream1.csv")
print(df_2['category'].value_counts())

