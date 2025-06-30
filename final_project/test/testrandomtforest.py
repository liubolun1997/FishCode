from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

import getDataset

X_train, X_test, y_train, y_test, X, y = getDataset.get_fridge_dateset()

X_train = X_train.fillna(X_train.mean())
smote = SMOTE(sampling_strategy='auto', random_state=42)
# smote = SMOTE(sampling_strategy={1: 20000, 2: 15000, 3: 15000}, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
