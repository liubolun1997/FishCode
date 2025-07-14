from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, GlobalMaxPooling1D
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import getDataset

X, y = getDataset.get_fridge_dateset()
input_layer = Input(shape=(X.shape[1], X.shape[2]))
x = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
x = GlobalMaxPooling1D()(x)  # 得到一个固定长度的向量

lstm_model = Model(inputs=input_layer, outputs=x)

# 提取特征
features = lstm_model.predict(X)

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# ==== 8. 评估模型 ====
y_pred = svm_model.predict(X_test)

print("\nSVM 分类报告:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, zero_division=0))