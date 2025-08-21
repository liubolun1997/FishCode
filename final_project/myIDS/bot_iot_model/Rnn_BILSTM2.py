import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")

# loading data
df = pd.read_csv('csvdataset/train1.csv')
df = df[~df['category'].isin(['Reconnaissance'])]
df = df.dropna()

# Features and tags
drop_cols = [
    'pkSeqID','stime','ltime','saddr','sport','daddr','dport',
    'category','subcategory','attack'
]

X = df.drop(columns=drop_cols)
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

X[cat_cols] = X[cat_cols].fillna('NaN')
X[num_cols] = X[num_cols].fillna(0)

encoders_rnn = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders_rnn[col] = le
joblib.dump(encoders_rnn, "./deploy_rnn/encoders_rnn.pkl")

y = df['category']

# label encoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Save the label encoder
joblib.dump(label_encoder, "./deploy_rnn/label_encoder_rnn.pkl")

# StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Save the standardizer
joblib.dump(scaler, "./deploy_rnn/scaler_rnn.pkl")

# Construct sequence data
time_steps = 5
n_samples = len(X_scaled) - time_steps
X_seq = np.array([X_scaled[i:i+time_steps] for i in range(n_samples)])
y_seq = y_onehot[time_steps:]

print(f"Enter the dimension: {X_seq.shape}, Label dimensions: {y_seq.shape}")

# Divide the training/test set
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build the RNN + BiLSTM model
model = Sequential()
model.add(SimpleRNN(64, return_sequences=True, input_shape=(time_steps, X_seq.shape[2])))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_seq.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# training model
history = model.fit(X_train, y_train, epochs=20, batch_size=64,
                    validation_data=(X_test, y_test), verbose=1)

# Save the model
model.save("./deploy_rnn/rnn_bilstm_model.h5")

# Model evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n Test accuracy: {accuracy * 100:.2f}%")

# Predict the outcome
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Output report
print("\n report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
