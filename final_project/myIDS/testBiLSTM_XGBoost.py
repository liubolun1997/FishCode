from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, GlobalMaxPooling1D
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import getDataset

X, y = getDataset.get_fridge_dateset()
input_layer = Input(shape=(X.shape[1], X.shape[2]))
x = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
x = GlobalMaxPooling1D()(x)  # 得到一个固定长度的向量

lstm_model = Model(inputs=input_layer, outputs=x)

# 提取特征
features = lstm_model.predict(X)

features, y = shuffle(features, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    objective='binary:logistic',
    base_score=0.5,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()