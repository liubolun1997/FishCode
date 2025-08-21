from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd

df_2 = pd.read_csv("csvdataset/val_stream1.csv")
print(df_2['type'].value_counts())