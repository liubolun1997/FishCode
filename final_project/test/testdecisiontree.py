import getDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

X_train, X_test, y_train, y_test, X, y = getDataset.get_fridge_dateset()
# training model
model = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Prediction accuracy：", accuracy_score(y_test, y_pred))
print("Detailed report：\n", classification_report(y_test, y_pred))