import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

# -------------------------
# LOAD DATA
# -------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_ids = test["id"]

# -------------------------
# SPLIT FEATURES / TARGET
# -------------------------
X = train.drop("Irrigation_Need", axis=1)
y = train["Irrigation_Need"]

# -------------------------
# ENCODE TARGET
# -------------------------
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# -------------------------
# HANDLE CATEGORICAL FEATURES
# -------------------------
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        combined = pd.concat([X[col], test[col]], axis=0).astype(str)
        le.fit(combined)

        X[col] = le.transform(X[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

# -------------------------
# MISSING VALUES
# -------------------------
X = X.fillna(X.mean())
test = test.fillna(test.mean())

# -------------------------
# SCALE DATA (optional but safe)
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# -------------------------
# MODEL + GRID SEARCH (THIS IS THE IMPORTANT PART)
# -------------------------
rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_scaled, y)

print("\nBEST PARAMETERS:")
print(grid.best_params_)

print("\nBEST CV SCORE:")
print(grid.best_score_)

best_model = grid.best_estimator_

# -------------------------
# CONFUSION MATRIX (TRAIN DATA CHECK)
# -------------------------
y_pred_train = best_model.predict(X_scaled)
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y, y_pred_train))

# -------------------------
# TRAIN FINAL MODEL (already done by GridSearch but safe)
# -------------------------
best_model.fit(X_scaled, y)

# -------------------------
# PREDICT TEST
# -------------------------
preds = best_model.predict(test_scaled)
preds = le_target.inverse_transform(preds)

# -------------------------
# SUBMISSION FILE
# -------------------------
submission = pd.DataFrame({
    "id": test_ids,
    "Irrigation_Need": preds
})

submission.to_csv("submission.csv", index=False)

print("\nDONE: submission.csv created")