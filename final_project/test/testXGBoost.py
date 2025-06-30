import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

import getDataset

X_train, X_test, y_train, y_test, X, y = getDataset.get_fridge_dateset()

# training XGBoost model
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, objective="multi:softmax", num_class=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Prediction accuracy：", accuracy_score(y_test, y_pred))
print("Detailed report：\n", classification_report(y_test, y_pred))

# X_train = X_train.fillna(X_train.mean())  # Fill with the mean
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# xgb = XGBClassifier(objective='multi:softmax', num_class=10, random_state=42)
#
# # Grid search parameter
# param_grid = {
#     'n_estimators': [100],
#     'max_depth': [5],
#     'learning_rate': [0.01],
#     'subsample': [0.8],
#     'colsample_bytree': [0.8],
#     'max_delta_step': [0]  # Prevent gradient explosion
# }

# # Use GridSearchCV to find the best parameters
# grid_search = GridSearchCV(xgb, param_grid, cv=3, verbose=1, scoring='accuracy')
# grid_search.fit(X_train_resampled, y_train_resampled)
#
# print("Best parameters:", grid_search.best_params_)

